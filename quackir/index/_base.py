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

from abc import ABC, abstractmethod
from quackir._base import IndexType

class Indexer(ABC):
    @abstractmethod
    def get_index_type(self, table_name: str) -> IndexType:
        pass

    @abstractmethod
    def get_num_rows(self, table_name: str) -> int:
        """Get the number of rows in the specified table."""
        pass

    def load_table(self, table_name: str, file_path: str, index_type: IndexType | None = None, pretokenized=False):
        """Load data into the specified table."""
        if index_type == None:
            index_type = self.get_index_type(table_name)
        if file_path.endswith('.jsonl'):
            self.load_jsonl_table(table_name, file_path, index_type, pretokenized)
        elif file_path.endswith('.parquet'):
            if index_type != IndexType.DENSE:
                raise ValueError("Loading parquet currently only supports dense indexes")
            self.load_parquet_table(table_name, file_path, index_type, pretokenized)
        else:
            return
        
        print(f"{self.get_num_rows(table_name)} rows loaded into {table_name} with {self.__class__.__name__}")

    @abstractmethod
    def load_parquet_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        pass

    @abstractmethod
    def load_jsonl_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        pass
    
    @abstractmethod
    def init_table(self, table_name: str, index_type: IndexType, embedding_dim=768): 
        pass
    
    @abstractmethod
    def fts_index(self, table_name: str = "corpus"):
        """Perform the indexing operation."""
        pass

    def close(self):
        self.conn.close()