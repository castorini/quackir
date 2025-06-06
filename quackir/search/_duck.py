import duckdb
from ._base import Searcher
from quackir._base import SearchType

class DuckDBSearcher(Searcher):
    def __init__(self, db_path="duck.db"):
        self.conn = duckdb.connect(db_path)

    def get_search_type(self, table_name: str) -> SearchType:
        table_description = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [row[0] for row in table_description]
        if "embedding" in column_names:
            return SearchType.DENSE
        elif "contents" in column_names:
            return SearchType.SPARSE
        else:
            raise ValueError(f"Unknown search type for table {table_name}. Ensure it has either an 'embedding' column or a 'contents' column.")

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

    def rrf_search(self, query_string, query_embedding, top_n=5, k=60, table_name=["sparse", "dense"]):
        sparse_table = table_name[0] if self.get_search_type(table_name[0]) == SearchType.SPARSE else table_name[1]
        dense_table = table_name[1] if self.get_search_type(table_name[1]) == SearchType.DENSE else table_name[0]
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
                ROW_NUMBER() OVER (ORDER BY COALESCE(fts_main_{sparse_table}.match_bm25(id, ?), 0) DESC) AS fts_rank
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
    
