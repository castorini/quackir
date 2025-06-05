from abc import ABC, abstractclassmethod
from pyserini.analysis import Analyzer, get_lucene_analyzer

class DB_Searcher(ABC):
    analyzer = Analyzer(get_lucene_analyzer())

    @staticmethod
    def tokenize(to_tokenize):
        return ' '.join(DB_Searcher.analyzer.analyze(to_tokenize))
    
    @staticmethod
    def filter_id(results, query_id):
        return [res for res in results if res[0] != query_id]
    
    def search(self, method, query_id, query_string, top_n=5):
        results = []
        if method == 'fts':
            results = self.fts_search(query_string, top_n=top_n)
        elif method == 'bge':
            results = self.embedding_search(query_id, top_n=top_n)
        elif method == 'rrf':
            results = self.rrf(query_id, query_string, top_n=top_n)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        return self.filter_id(results, query_id)

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