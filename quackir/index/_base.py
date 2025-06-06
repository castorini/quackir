from abc import ABC, abstractmethod
from quackir._base import IndexType

class Indexer(ABC):
    @staticmethod
    def count_lines(filename):
       with open(filename, 'r') as file:
           return sum(1 for _ in file)

    @abstractmethod
    def init_table(self, table_name: str, file_path: str, index_type: IndexType, pretokenized=False, embedding_dim=768): 
        """Initialize the database tables for indexing."""
        pass

    @abstractmethod
    def fts_index(self, table_name: str = "corpus"):
        """Perform the indexing operation."""
        pass