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

from ragdb import RAGdb
from ._base import Searcher
from quackir._base import SearchType


class RagDBSearcher(Searcher):
    def __init__(self, db_path: str = "quackir.ragdb"):
        self.conn = RAGdb(db_path)

    def get_search_type(self, table_name: str) -> SearchType:
        return SearchType.SPARSE

    def fts_search(self, query_string: str, top_n=5, table_name="corpus"):  # type: ignore[override]
        hits = self.conn.search(query_string, top_k=top_n) or []
        return [
            (
                getattr(h, "path"),
                getattr(h, "score"),
            )
            for h in hits
        ]

    def embedding_search(self, query_embedding, top_n=5, table_name="corpus"):  # type: ignore[override]
        raise NotImplementedError("Dense embedding search not supported via RAGdb API.")

    def rrf_search(  # type: ignore[override]
        self,
        query_string: str,
        query_embedding,
        top_n=5,
        k=60,
        table_names=["sparse", "dense"],
    ):
        raise NotImplementedError(
            "RRF search is not supported via the RAGdb API since dense embeddings are not supported."
        )
