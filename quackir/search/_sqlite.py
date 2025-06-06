import sqlite3
from ._base import DBSearcher

class SQLiteSearcher(DBSearcher):
    def __init__(self, db_path="sqlite.db"):
        self.conn = sqlite3.connect(db_path)

    def fts_search(self, query_string, top_n=5, table_name="corpus"):
        query = f"""
        SELECT id, bm25(fts_{table_name})*-1 AS score
        FROM fts_{table_name}
        WHERE fts_{table_name} MATCH '{{}}'
        ORDER BY score DESC
        LIMIT {{}}
        """
        query_string = query_string.replace("'", "''").replace('"', '""')
        terms = query_string.split()
        escaped_terms = terms
        # make each term a string so that any special chars are escaped
        escaped_terms = [f'"{term}"' for term in escaped_terms]
        # allow matching of any of the terms, using + or AND will turn it into boolean AND retrieval
        res = " OR ".join(escaped_terms)
        query = query.format(res, top_n)
        return self.conn.execute(query).fetchall()
    
    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus"):
        pass

    def rrf_search(self, query_string: str, query_embedding: str, top_n=5, k=60, table_name="corpus"):
        pass