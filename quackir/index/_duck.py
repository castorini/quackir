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
from quackir.flock import FlockManager
import duckdb
import json

class DuckDBIndexer(Indexer):
    def __init__(self, db_path="duck.db", flock_manager: FlockManager | None = None):
        self.conn = duckdb.connect(db_path)
        self.flock_manager = flock_manager

    def get_index_type(self, table_name: str) -> IndexType:
        table_description = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_names = [row[0] for row in table_description]
        if "contents" in column_names:
            return IndexType.SPARSE
        elif "embedding" in column_names:
            return IndexType.DENSE
        else:
            raise ValueError(f"Unknown index type for table {table_name}. Ensure it has either an 'embedding' column or a 'contents' column.")

    def load_table(
        self,
        table_name: str,
        file_path: str,
        index_type: IndexType | None = None,
        pretokenized=False,
        with_flock=False,
        **flock_kwargs,
    ):
        """
        Load a file into a table.

        If with_flock=True, build a dense embedding table directly from the source file
        using the Flock extension instead of loading raw contents.

        Extra keyword arguments are forwarded to create_embedding, e.g.:
            id_column="id", contents_column="contents", embedding_dim=768
        """
        if with_flock:
            if self.flock_manager is None:
                raise ValueError("flock_manager must be provided when with_flock=True.")
            self.flock_manager.create_embedding(
                dest_table=table_name, file_path=file_path, **flock_kwargs
            )
        else:
            super().load_table(table_name, file_path, index_type, pretokenized)

    def init_table(self, table_name: str, index_type: IndexType, embedding_dim=768):
        self.conn.execute(f"""DROP TABLE IF EXISTS {table_name}""")
        if index_type == IndexType.SPARSE:
            self.conn.execute(f"""CREATE TABLE {table_name} (id VARCHAR, contents VARCHAR)""")
        elif index_type == IndexType.DENSE:
            self.conn.execute(f"""CREATE TABLE {table_name} (id VARCHAR, embedding DOUBLE[{embedding_dim}])""")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
    def load_jsonl_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        rows = [[d["id"], d["contents"] if "contents" in d else d["vector"]] for d in data]
        if index_type == IndexType.SPARSE and not pretokenized:
            rows = [[d["id"], tokenize(d["contents"])] for d in data]
        if index_type == IndexType.SPARSE:
            self.conn.executemany(f"insert into {table_name} (id, contents) values (?, ?)", rows)
        elif index_type == IndexType.DENSE:
            self.conn.executemany(f"""insert into {table_name} (id, embedding) values (?, ?)""", rows)
 
    def load_parquet_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        column_names = self.conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").fetchall()
        self.conn.execute(f"""INSERT INTO {table_name} SELECT {column_names[0][0]} as id, {column_names[1][0]} as embedding FROM read_parquet('{file_path}')""")

    def get_num_rows(self, table_name: str) -> int:
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def fts_index(self, table_name: str = "corpus"):
        self.conn.execute(f"PRAGMA create_fts_index({table_name}, id, contents, stemmer = 'none', stopwords = 'none', ignore = 'a^', strip_accents = 0, lower = 0, overwrite = 1)")