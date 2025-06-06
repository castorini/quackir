from ._base import DBIndexer
from quackir._base import IndexType
from quackir.analysis import tokenize
import sqlite3
from tqdm import tqdm
import json

class SQLiteIndexer(DBIndexer):
    def __init__(self, db_path="sqlite.db"):
        self.conn = sqlite3.connect(db_path)

    def init_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False, embedding_dim=768): 
        if index_type != IndexType.SPARSE:
            raise ValueError(f"SQLite only supports FTS indexing, got {index_type}")
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"""
            CREATE TABLE {table_name} (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        """)

        num_lines = self.count_lines(file_path)
        with open(file_path, "r") as file:
            for line in tqdm(file, total=num_lines, desc=f"Loading {table_name}"):
                row = json.loads(line.strip())
                if not pretokenized:
                    contents = tokenize(row['contents'])
                else:
                    contents = row['contents']
                self.conn.execute(
                    f"""
                    INSERT INTO {table_name} (id, contents) VALUES (?, ?)
                """,
                    (row["id"], contents),
                )
        num_rows = self.conn.execute(f"select count(*) from {table_name}").fetchone()
        print(f"Loaded {num_rows[0]} rows into {table_name}")

    def fts_index(self, table_name: str = "corpus"):
        self.conn.execute(f"drop table if exists fts_{table_name}")
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE fts_{table_name} USING fts5(
                id, contents,
                content='{table_name}', content_rowid='rowid', tokenize = 'porter' 
            )
        """)

        self.conn.execute(
            f"INSERT INTO fts_{table_name} (id, contents) SELECT id, contents FROM {table_name};"
        )
        self.conn.commit()
        