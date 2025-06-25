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

from quackir._base import SearchType
from quackir.analysis import tokenize
from abc import ABC, abstractmethod

class Searcher(ABC):
    @staticmethod
    def filter_id(results, query_id):
        return [res for res in results if res[0] != query_id]

    def search(self, method: SearchType, query_id: str = None, query_string: str = None, query_embedding: str = None, top_n=5, tokenize_query=True, table_names: list =["corpus"], rrf_k=60):
        results = []
        if method != SearchType.DENSE and tokenize_query:
            query_string = tokenize(query_string)
        if method == SearchType.SPARSE:
            results = self.fts_search(query_string, top_n=top_n, table_name=table_names[0])
        elif method == SearchType.DENSE:
            results = self.embedding_search(query_embedding, top_n=top_n, table_name=table_names[0])
        elif method == SearchType.HYBRID:
            results = self.rrf_search(query_string, query_embedding, top_n=top_n, k=rrf_k, table_names=table_names)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        return self.filter_id(results, query_id)
    
    @abstractmethod
    def get_search_type(self, table_name: str) -> SearchType:
        pass
    
    @abstractmethod
    def fts_search(self, query_string: str, top_n=5, table_name="corpus"):
        pass

    @abstractmethod
    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus"):
        pass

    @abstractmethod
    def rrf_search(self, query_string: str, query_embedding: str, top_n=5, k=60, table_names=["sparse", "dense"]):
        pass

    def close(self):
        self.conn.close()