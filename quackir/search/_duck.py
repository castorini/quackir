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

import duckdb
from ._base import Searcher
from quackir._base import SearchType
from quackir.flock import FlockManager

class DuckDBSearcher(Searcher):
    def __init__(self, db_path="duck.db", flock_manager: FlockManager | None = None):
        self.conn = duckdb.connect(db_path)
        self.flock_manager = flock_manager

    def get_search_type(self, table_name: str) -> SearchType:
        table_description = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [row[0] for row in table_description]
        if "contents" in column_names:
            return SearchType.SPARSE
        elif "embedding" in column_names:
            return SearchType.DENSE
        else:
            raise ValueError(f"Unknown search type for table {table_name}. Ensure it has either an 'embedding' column or a 'contents' column.")

    def fts_search(self, query_string, top_n=5, table_name="corpus"):
        query = f"""
        WITH fts AS (
            SELECT *, COALESCE(fts_main_{table_name}.match_bm25(id, ?, k:=0.9, b:=0.4), 0) AS score
            FROM {table_name}
        )
        SELECT id, score
        FROM fts
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT {top_n};
        """
        return self.conn.execute(query, [query_string]).fetchall()

    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus", with_flock=False, **flock_kwargs):
        """
        Perform dense retrieval.

        If with_flock=True, the query embedding is computed via the Flock extension before searching.

        Extra keyword arguments are forwarded to search_embedding, e.g.:
            embedding_dim=768
        """
        if with_flock:
            if self.flock_manager is None:
                raise ValueError("flock_manager must be provided when with_flock=True.")
            return self.flock_manager.search_embedding(
                query_text=query_embedding,
                top_n=top_n,
                table_name=table_name,
                **flock_kwargs,
            )
        embd_size = len(query_embedding)
        query_embedding = str(query_embedding)
        query = f"""
        WITH query_embedding AS (
            SELECT array{query_embedding}::DOUBLE[{embd_size}] AS embedding
        )
        SELECT {table_name}.id, 
            array_cosine_similarity({table_name}.embedding, query_embedding.embedding) AS score
        FROM {table_name}, query_embedding
        ORDER BY score DESC
        LIMIT {top_n}
        """
        return self.conn.execute(query).fetchall()

    def rrf_search(self, query_string, query_embedding, top_n=5, k=60, table_names=["sparse", "dense"]):
        sparse_table = table_names[0] if self.get_search_type(table_names[0]) == SearchType.SPARSE else table_names[1]
        dense_table = table_names[1] if self.get_search_type(table_names[1]) == SearchType.DENSE else table_names[0]
        embd_size = len(query_embedding)
        query_embedding = str(query_embedding)
        query = f"""
        WITH 
        query_embedding AS (
            SELECT array{query_embedding}::DOUBLE[{embd_size}] AS embedding
        ),
        embd AS (
            SELECT {dense_table}.id,
                ROW_NUMBER() OVER (ORDER BY array_cosine_similarity({dense_table}.embedding, query_embedding.embedding) DESC) AS sim_rank
            FROM {dense_table}, query_embedding
            limit {top_n}
        ),
        fts AS (
            SELECT id, 
                ROW_NUMBER() OVER (ORDER BY COALESCE(fts_main_{sparse_table}.match_bm25(id, ?, k:=0.9, b:=0.4), 0) DESC) AS fts_rank
            FROM {sparse_table}
            limit {top_n}
        ),
        combined_results AS (
            SELECT 
                COALESCE(s.id, f.id) AS id,
                COALESCE(1.0 / ({k} + s.sim_rank), 0) + COALESCE(1.0 / ({k} + f.fts_rank), 0) AS rrf_score
            FROM embd s
            FULL OUTER JOIN fts f ON s.id = f.id
        )
        SELECT id, rrf_score
        FROM combined_results
        ORDER BY rrf_score DESC
        LIMIT {top_n}
        """
        return self.conn.execute(query, [query_string]).fetchall()
    
