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
import json
import warnings
from ragdb import RAGdb
from ._base import Indexer
from quackir._base import IndexType


class RagDBIndexer(Indexer):
    def __init__(self, db_path: str = "quackir.ragdb"):
        self.conn = RAGdb(db_path)

    def get_index_type(self, table_name: str) -> IndexType:
        return IndexType.SPARSE

    def init_table(self, table_name: str, index_type: IndexType, embedding_dim=768):
        if index_type == IndexType.DENSE:
            raise NotImplementedError(
                "Dense vector indexing not supported via RAGdb API."
            )
        return

    def load_jsonl_table(
        self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False
    ):
        if index_type == IndexType.SPARSE:
            staging = os.path.join(
                os.path.dirname(file_path), f"ragdb_staging_{table_name}"
            )
            os.makedirs(staging, exist_ok=True)
            with open(file_path, "r") as f:
                for line in f:
                    d = json.loads(line)
                    doc_id = d["id"]
                    contents = d.get("contents", "")
                    out_path = os.path.join(staging, f"{doc_id}.txt")
                    with open(out_path, "w", encoding="utf-8") as out:
                        out.write(contents)
            self.conn.ingest_folder(staging)
        else:
            raise NotImplementedError(
                "Dense vector indexing not supported via RAGdb API."
            )

    def load_parquet_table(
        self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False
    ):
        raise NotImplementedError("Parquet ingestion not supported for RAGdb adapter.")

    def get_num_rows(self, table_name: str) -> int:
        docs = self.conn.list_documents(limit=10**9)
        return len(docs)

    def fts_index(self, table_name: str = "corpus"):
        warnings.warn(
            "RagDBIndexer.fts_index is a no-op: RagDB builds its TF-IDF vectors during ingest and `_rebuild_vectors`.",
            RuntimeWarning,
        )
        return
