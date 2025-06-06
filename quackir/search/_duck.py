import duckdb
from ._base import DBSearcher
import numpy as np

class DuckDBSearcher(DBSearcher):
    def __init__(self, db_path="duck.db"):
        self.conn = duckdb.connect(db_path)

    def fts_search(self, query_string, top_n=5, table_name="corpus"):
        query = f"""
        WITH fts AS (
            SELECT *, COALESCE(fts_main_{table_name}.match_bm25(id, ?, k:=0.9, b:=0.4), 0) AS score
            FROM corpus
        )
        SELECT id, score
        FROM fts
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT {top_n};
        """
        return self.conn.execute(query, [query_string]).fetchall()
    
    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus"):
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

    def rrf_search(self, query_string, query_embedding, top_n=5, k=60, table_name="corpus"):
        embd_size = len(query_embedding)
        query_embedding = str(query_embedding)
        query = f"""
        WITH 
        query_embedding AS (
            SELECT array{query_embedding}::DOUBLE[{embd_size}] AS embedding
        ),
        embd AS (
            SELECT {table_name}.id,
                ROW_NUMBER() OVER (ORDER BY array_cosine_similarity({table_name}.embedding, query_embedding.embedding) DESC) AS sim_rank
            FROM {table_name}, query_embedding
            limit {top_n}
        ),
        fts AS (
            SELECT id, 
                ROW_NUMBER() OVER (ORDER BY COALESCE(fts_main_{table_name}.match_bm25(id, ?), 0) DESC) AS fts_rank
            FROM {table_name}
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
    
