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

import os
import config
from ._base import Indexer
from quackir._base import IndexType
from quackir.analysis import tokenize
from quackir.utils.constants import BM25_INDEX_TEMPLATE, EMBEDDING_INDEX_TEMPLATE
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from io import StringIO
import json

class PostgresIndexer(Indexer):
    def __init__(
        self, db_name="quackir", user="postgres", use_pg_textsearch: bool = False
    ):
        self.use_pg_textsearch = use_pg_textsearch
        if self.use_pg_textsearch:
            self.conn = psycopg2.connect(dsn=os.environ["TIMESCALE_SERVICE_URL"])
            cur = self.conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
        else:
            self.conn = psycopg2.connect(dbname=db_name, user=user)

    def get_index_type(self, table_name: str) -> IndexType:
        cur = self.conn.cursor()
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
        columns = [row[0] for row in cur.fetchall()]
        if "contents" in columns:
            return IndexType.SPARSE
        elif "embedding" in columns:
            return IndexType.DENSE
        else:
            raise ValueError(f"Unknown index type for table {table_name}. Ensure it has either an 'embedding' column or a 'contents' column.")

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
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        # postgres does not allow null characters
        rows = [(d["id"], d["contents"].replace("\x00", "\uFFFD") if "contents" in d else d["vector"]) for d in data]
        if index_type == IndexType.SPARSE and not pretokenized:
            rows = [(d["id"], tokenize(d["contents"])) for d in data]
        cur = self.conn.cursor()
        if index_type == IndexType.SPARSE:
            execute_values(cur, f"INSERT INTO {table_name} (id, contents) VALUES %s", rows)
        elif index_type == IndexType.DENSE:
            execute_values(cur, f"INSERT INTO {table_name} (id, embedding) VALUES %s", rows)
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

    def fts_index(
        self,
        table_name: str = "corpus",
        text_config: str = "english",
        k1: float = 0.9,
        b: float = 0.4,
    ):
        cur = self.conn.cursor()
        if self.use_pg_textsearch:
            idx_name = BM25_INDEX_TEMPLATE.format(table_name=table_name)
            cur.execute(f'DROP INDEX IF EXISTS "{idx_name}";')
            cur.execute(
                f"""
                CREATE INDEX "{idx_name}" ON "{table_name}"
                USING bm25(contents) WITH (text_config='{text_config}', k1={k1}, b={b});
                """
            )
        else:
            cur.execute(
                f"""
                CREATE INDEX "corpus_contents_gin" ON "{table_name}"
                USING gin(to_tsvector('simple', contents));
                """
            )
        self.conn.commit()

    def embedding_index(
        self,
        table_name: str,
        using: str = "hnsw",
        opclass: str = "vector_cosine_ops",
        column: str = "embedding",
    ) -> str:
        idx_name = EMBEDDING_INDEX_TEMPLATE.format(table_name=table_name)
        cur = self.conn.cursor()
        cur.execute(f'DROP INDEX IF EXISTS "{idx_name}";')
        cur.execute(
            f"""
            CREATE INDEX "{idx_name}" ON "{table_name}"
            USING {using} ({column} {opclass});
            """
        )
        self.conn.commit()
        return idx_name
