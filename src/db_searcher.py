from abc import ABC, abstractclassmethod
from pyserini.analysis import Analyzer, get_lucene_analyzer

class DB_Searcher(ABC):
    analyzer = Analyzer(get_lucene_analyzer())

    def tokenize(self, to_tokenize):
        return ' '.join(self.analyzer.analyze(to_tokenize))

    @abstractclassmethod
    def init_tables(self, table_name: str, file_path: str, method: str, pretokenized=False):
        pass

    @abstractclassmethod
    def fts_search(self, query_string: str, top_n=5, tokenize_query=False):
        pass

    @abstractclassmethod
    def fts_index(self):
        pass

    @abstractclassmethod
    def embedding_search(self, query_id: str, top_n=5):
        pass

    @abstractclassmethod
    def rrf(self, query_id: str, query_string: str, top_n=5, k=60):
        pass

    @abstractclassmethod
    def get_queries(self):
        pass