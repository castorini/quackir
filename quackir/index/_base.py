from abc import ABC, abstractmethod
from quackir._base import IndexType

class Indexer(ABC):
    @abstractmethod
    def get_num_rows(self, table_name: str) -> int:
        """Get the number of rows in the specified table."""
        pass

    def load_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False):
        """Load data into the specified table."""
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