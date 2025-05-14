import sqlite3
import json
from db_searcher import DB_Searcher

class SQLite_Searcher(DB_Searcher):
    def __init__(self):
        self.conn = sqlite3.connect("sqlite.db")

    def init_tables(self, table_name: str, file_path: str, method: str):
        self.conn.execute(f"drop table if exists {table_name}")
        self.conn.execute(f"""
            CREATE TABLE {table_name} (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        """)
        with open(file_path, "rt", encoding="utf-8") as file:
            for line in file:
                row_data = json.loads(line.strip())
                self.conn.execute(
                    f"""
                    INSERT INTO {table_name} (id, contents) VALUES (?, ?)
                """,
                    (row_data["id"], row_data["contents"]),
                )
        num_rows = self.conn.execute(f"select count(*) from {table_name}").fetchone()
        print(f"Loaded {num_rows[0]} rows in {table_name}")

    def fts_search(self, query_string, top_n=5):
        query = """
        SELECT id, bm25(fts_corpus)*-1 AS score
        FROM fts_corpus
        WHERE fts_corpus MATCH '{}'
        ORDER BY score DESC
        LIMIT {}
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

    def fts_index(self):
        self.conn.execute("drop table if exists fts_corpus")
        self.conn.execute("""
            CREATE VIRTUAL TABLE fts_corpus USING fts5(
                id, contents,
                content='corpus', content_rowid='rowid', tokenize = 'porter' 
            )
        """)

        self.conn.execute(
            "INSERT INTO fts_corpus (id, contents) SELECT id, contents FROM corpus;"
        )
        self.conn.commit()

    def get_queries(self):
        return self.conn.execute("SELECT id, contents FROM query").fetchall()
    
    def rrf(self, query_id: str, query_string: str, top_n=5, k=60):
        pass

    def embedding_search(self, query_id: str, top_n=5):
        pass