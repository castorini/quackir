import psycopg2
import re
from ._base import Searcher
from quackir._base import SearchType

class PostgresSearcher(Searcher):
    def __init__(self, db_name="quackir", user="postgres"):
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
        cur = self.conn.cursor()
        cur.execute(query, (ts_query, ts_query, top_n))
        return cur.fetchall()
    
    def embedding_search(self, query_embedding, top_n=5, table_name="corpus"):
        cur = self.conn.cursor()
        query = f"""select id, 1 - (embedding <=> %s::vector) as score from {table_name} order by score desc limit %s"""
        cur.execute(query, (query_embedding, top_n))
        return cur.fetchall()
    
    def rrf_search(self, query_string: str, query_embedding: str, top_n=5, k=60, table_name=["sparse", "dense"]):
        sparse_table = table_name[0] if self.get_search_type(table_name[0]) == SearchType.SPARSE else table_name[1]
        dense_table = table_name[1] if self.get_search_type(table_name[1]) == SearchType.DENSE else table_name[0]
        ts_query = self.clean_tsquery(query_string)
        cur = self.conn.cursor()
        sql = f"""
        WITH semantic_search AS (
            SELECT id, RANK () OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank
            FROM {dense_table}
            LIMIT %(n)s
        ),
        keyword_search AS (
            SELECT id, RANK () OVER (ORDER BY ts_rank(to_tsvector('simple', contents), query) DESC) as rank
            FROM {sparse_table}, to_tsquery('simple', %(query)s) query
            WHERE to_tsvector('simple', contents) @@ query
            LIMIT %(n)s
        )
        SELECT
            COALESCE(semantic_search.id, keyword_search.id) AS id,
            COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
        FROM semantic_search
        FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
        ORDER BY score DESC
        LIMIT %(n)s
        """
        cur.execute(sql, {'query': ts_query, 'vector': query_embedding, 'n': top_n, 'k': k})
        results = cur.fetchall()
        return results