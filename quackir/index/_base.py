from abc import ABC, abstractmethod

class DBIndexer(ABC):
    @staticmethod
    def count_lines(filename):
       with open(filename, 'r') as file:
           return sum(1 for _ in file)

    @abstractmethod
    def init_table(self, table_name: str, file_path: str, index_type: str, pretokenized=False, embedding_dim=768): 
        """Initialize the database tables for indexing."""
        pass

    @abstractmethod
    def fts_index(self, table_name: str = "corpus"):
        """Perform the indexing operation."""
        pass