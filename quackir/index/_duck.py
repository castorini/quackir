from ._base import DBIndexer
from quackir.common.enums import SearchType
from quackir.analysis import tokenize
import duckdb
from tqdm import tqdm
import json

class DuckDBIndexer(DBIndexer):
    def __init__(self, db_path="duck.db"):
        self.conn = duckdb.connect(db_path)

    def init_table(self, table_name: str, file_path: str, index_type: SearchType, pretokenized=False, embedding_dim=768):
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        if index_type == SearchType.FTS:
            self.conn.execute(f"""CREATE TABLE {table_name} (id varchar primary key, contents varchar)""")
        elif index_type == SearchType.EMBD:
            self.conn.execute(f"""CREATE TABLE {table_name} (id varchar primary key, embedding double[{embedding_dim}])""")
        elif index_type == SearchType.RRF:
            self.conn.execute(f"""CREATE TABLE {table_name} (id varchar primary key, contents varchar, embedding double[{embedding_dim}])""")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        num_lines = self.count_lines(file_path)
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=num_lines, desc=f"Loading {table_name}"):
                row = json.loads(line.strip())
                if not pretokenized and index_type != SearchType.EMBD:
                    contents = tokenize(row['contents'])
                else:
                    contents = row['contents']
                if index_type == SearchType.FTS:
                    self.conn.execute(f"insert into {table_name} (id, contents) values (?, ?)", (row['id'], contents))
                elif index_type == SearchType.EMBD:
                    self.conn.execute(f"""insert into {table_name} (id, embedding) values (?, ?)""", (row['id'], row['vector']))
                elif index_type == SearchType.RRF:
                    self.conn.execute(f"""insert into {table_name} (id, contents, embedding) values (?, ?, ?)""", (row['id'], contents, row['vector']))

        self.conn.execute(f"select count(*) from {table_name}")
        print(f"Loaded {self.conn.fetchone()[0]} rows into {table_name}")

    def fts_index(self, table_name: str = "corpus"):
        self.conn.execute(f"PRAGMA create_fts_index({table_name}, id, contents, stemmer = 'none', stopwords = 'none', ignore = 'a^', strip_accents = 0, lower = 0, overwrite = 1)")