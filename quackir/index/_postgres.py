from ._base import Indexer
from quackir._base import IndexType
from quackir.analysis import tokenize
import psycopg2
import pandas as pd
from io import StringIO

class PostgresIndexer(Indexer):
    def __init__(self, db_name="quackir", user="postgres"):
        self.conn = psycopg2.connect(dbname=db_name, user=user)

    def init_table(self, table_name: str, index_type: IndexType, embedding_dim=768):
        cur = self.conn.cursor()
        cur.execute(f"drop table if exists {table_name}")  
        if index_type == IndexType.SPARSE:
            cur.execute(f"create table {table_name} (id text primary key, contents text);")
        elif index_type == IndexType.DENSE:
            cur.execute(f"create table {table_name} (id text primary key, embedding vector({embedding_dim}));")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        self.conn.commit()

    def load_jsonl_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        df = pd.read_json(file_path, lines=True)
        if index_type == IndexType.SPARSE and not pretokenized:
            df["contents"] = df["contents"].apply(tokenize)
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cur = self.conn.cursor()
        cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)
        self.conn.commit()

    @staticmethod
    def format_vector_for_pg(v):
        return f"[{', '.join(f'{x:.8f}' for x in v)}]"

    def load_parquet_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        df = pd.read_parquet(file_path)
        df["vector"] = df["vector"].apply(self.format_vector_for_pg)
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cur = self.conn.cursor()
        cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buffer)
        self.conn.commit()

    def get_num_rows(self, table_name: str) -> int:
        cur = self.conn.cursor()
        cur.execute(f"select count(*) from {table_name}")
        result = cur.fetchone()
        return result[0] if result else 0

    def fts_index(self, table_name: str = "corpus"):
        cur = self.conn.cursor()
        cur.execute(f'''CREATE INDEX "corpus_contents_gin" ON "{table_name}" USING gin(to_tsvector('simple', contents));''')