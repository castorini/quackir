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

from ._base import Indexer
from quackir._base import IndexType
from quackir.analysis import tokenize
import sqlite3
import json

class SQLiteIndexer(Indexer):
    def __init__(self, db_path="sqlite.db"):
        self.conn = sqlite3.connect(db_path)

    def get_index_type(self, table_name: str) -> IndexType:
        cur = self.conn.execute(f'select * from {table_name}')
        names = list(map(lambda x: x[0], cur.description))
        if "contents" in names:
            return IndexType.SPARSE
        else:
            raise ValueError(f"Unknown index type for table {table_name}. Ensure it has a 'contents' column. SQLite only supports sparse indexes currently.")

    def init_table(self, table_name: str, index_type: IndexType, embedding_dim=768):
        if index_type != IndexType.SPARSE:
            raise ValueError(f"SQLite only supports FTS indexing, got {index_type}")
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.execute(f"""
            CREATE TABLE {table_name} (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        """)

    def load_parquet_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        pass

    def load_jsonl_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        if index_type != IndexType.SPARSE:
            raise ValueError("Sorry, SQLite indexing currently only supports the sparse method.")
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        rows = [(d["id"], d["contents"]) for d in data]
        if not pretokenized:
            rows = [(d["id"], tokenize(d["contents"])) for d in data]
        self.conn.executemany(f"INSERT INTO {table_name} (id, contents) VALUES (?, ?)", rows)
        self.conn.commit()
    
    def get_num_rows(self, table_name: str) -> int:
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

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
        