from ._base import DBIndexer
from quackir.common.enums import SearchType
from quackir.analysis import tokenize
import psycopg2
from tqdm import tqdm
import json

class PostgresIndexer(DBIndexer):
    def __init__(self, db_name="quackir", user="postgres"):
        self.conn = psycopg2.connect(dbname=db_name, user=user)

    def init_table(self, table_name: str, file_path: str, index_type: str, pretokenized=False, embedding_dim=768): 
        cur = self.conn.cursor()
        cur.execute(f"drop table if exists {table_name}")

        if index_type == SearchType.FTS:
            cur.execute(f"create table {table_name} (id text, contents text);")
        elif index_type == SearchType.EMBD:
            cur.execute(f"create table {table_name} (id text, embedding vector({embedding_dim}));")
        elif index_type == SearchType.RRF:
            cur.execute(f"create table {table_name} (id text, contents text, embedding vector({embedding_dim}));")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        num_lines = self.count_lines(file_path)
        with open(file_path, 'r') as f:
            for line in tqdm(f, total=num_lines, desc=f"Loading {table_name}"):
                row = json.loads(line)
                if not pretokenized and index_type != SearchType.EMBD:
                    contents = tokenize(row['contents'])
                else:
                    contents = row['contents']
                if index_type == SearchType.FTS:
                    cur.execute(f"insert into {table_name} (id, contents) values (%s, %s)", (row['id'], contents))
                elif index_type == SearchType.EMBD:
                    cur.execute(f"insert into {table_name} (id, embedding) values (%s, %s)", (row['id'], row['vector']))
                elif index_type == SearchType.RRF:
                    cur.execute(f"insert into {table_name} (id, contents, embedding) values (%s, %s, %s)", (row['id'], contents, row['vector']))
        
        self.conn.commit()
        cur.execute(f"select count(*) from {table_name}")
        print(f"Loaded {cur.fetchone()[0]} rows into {table_name}")

    def fts_index(self, table_name: str = "corpus"):
        cur = self.conn.cursor()
        cur.execute(f'''CREATE INDEX "corpus_contents_gin" ON "{table_name}" USING gin(to_tsvector('simple', contents));''')