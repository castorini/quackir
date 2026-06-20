#
# QuackIR: Reproducible IR research in RDBMS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import config
import psycopg2
import re
from ._base import Searcher
from quackir._base import SearchType
from quackir.utils.constants import BM25_INDEX_TEMPLATE
from psycopg2 import sql

class PostgresSearcher(Searcher):
    def __init__(
        self, db_name="quackir", user="postgres", use_pg_textsearch: bool = False
    ):
        self.use_pg_textsearch = use_pg_textsearch
        if self.use_pg_textsearch:
            self.conn = psycopg2.connect(dsn=os.environ["TIMESCALE_SERVICE_URL"])
            cur = self.conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
        else:
            self.conn = psycopg2.connect(dbname=db_name, user=user)

    @staticmethod
    def clean_tsquery(query_string):
        cleaned_query = re.sub(r'[^\w\s]', ' ', query_string)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        ts_query = " | ".join(cleaned_query.split())
        return ts_query
    
    def get_search_type(self, table_name: str) -> SearchType:
        cur = self.conn.cursor()
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
        columns = [row[0] for row in cur.fetchall()]
        if "contents" in columns:
            return SearchType.SPARSE
        elif "embedding" in columns:
            return SearchType.DENSE
        else:
            raise ValueError(f"Unknown search type for table {table_name}. Ensure it has either an 'embedding' column or a 'contents' column.")
    
    def fts_search(self, query_string, top_n=5, table_name="corpus"):
        cur = self.conn.cursor()
        if self.use_pg_textsearch:
            idx_name = BM25_INDEX_TEMPLATE.format(table_name=table_name)
            query = f"""
                    SELECT id,
                        contents <@> to_bm25query(%(q)s, %(idx)s) AS score
                    FROM "{table_name}"
                    ORDER BY score
                    LIMIT %(n)s
                    """
            cur.execute(query, {"q": query_string, "idx": idx_name, "n": top_n})
        else:
            ts_query = self.clean_tsquery(query_string)
            query = f"""
                    SELECT 
                        id, 
                        ts_rank(to_tsvector('simple', contents), to_tsquery('simple', %s)) AS score
                    FROM 
                        {table_name}
                    WHERE
                        to_tsvector('simple', contents) @@ to_tsquery('simple', %s)
                    ORDER BY 
                        score DESC
                    LIMIT %s
                    """
            cur.execute(query, (ts_query, ts_query, top_n))
        return cur.fetchall()
    
    def embedding_search(self, query_embedding, top_n=5, table_name="corpus"):
        cur = self.conn.cursor()
        query = f"""select id, 1 - (embedding <=> %s::vector) as score from {table_name} order by score desc limit %s"""
        cur.execute(query, (query_embedding, top_n))
        return cur.fetchall()
    
    def rrf_search(
        self,
        query_string: str,
        query_embedding: str,
        top_n=5,
        k=60,
        table_names=["sparse", "dense"],
        weight_keyword: float = 1.0,
        weight_semantic: float = 1.0,
    ):
        sparse_table = (
            table_names[0]
            if self.get_search_type(table_names[0]) == SearchType.SPARSE
            else table_names[1]
        )
        dense_table = (
            table_names[1]
            if self.get_search_type(table_names[1]) == SearchType.DENSE
            else table_names[0]
        )
        cur = self.conn.cursor()

        params = {
            "vector": query_embedding,
            "n": top_n,
            "k": k,
            "w_sem": weight_semantic,
            "w_key": weight_keyword,
        }

        vector_cte = f"""
            SELECT id,
                ROW_NUMBER() OVER (ORDER BY embedding <=> %(vector)s::vector, id) AS rank
            FROM "{dense_table}"
            ORDER BY rank
            LIMIT %(n)s
        """

        if self.use_pg_textsearch:
            idx_name = BM25_INDEX_TEMPLATE.format(table_name=sparse_table)
            params.update({"q": query_string, "idx": idx_name})
            keyword_cte = f"""
                SELECT id,
                    ROW_NUMBER() OVER (
                        ORDER BY contents <@> to_bm25query(%(q)s, %(idx)s), id
                    ) AS rank
                FROM "{sparse_table}"
                ORDER BY rank
                LIMIT %(n)s
            """
        else:
            ts_query = self.clean_tsquery(query_string)
            params["q"] = ts_query
            keyword_cte = f"""
                SELECT id,
                    ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('simple', contents), query) DESC, id) as rank
                FROM "{sparse_table}", to_tsquery('simple', %(q)s) query
                WHERE to_tsvector('simple', contents) @@ query
                ORDER BY rank
                LIMIT %(n)s
            """

        query_sql = f"""
            WITH vector_search AS (
                {vector_cte}
            ),
            keyword_search AS (
                {keyword_cte}
            )
            SELECT COALESCE(v.id, k.id) AS id,
                %(w_sem)s * COALESCE(1.0 / (%(k)s + v.rank), 0.0) +
                %(w_key)s * COALESCE(1.0 / (%(k)s + k.rank), 0.0) AS score
            FROM vector_search v
            FULL OUTER JOIN keyword_search k ON v.id = k.id
            ORDER BY score DESC
            LIMIT %(n)s
        """

        cur.execute(query_sql, params)
        return cur.fetchall()