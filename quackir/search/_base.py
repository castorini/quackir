from quackir._base import SearchType
from quackir.analysis import tokenize
from abc import ABC, abstractclassmethod

class DBSearcher(ABC):
    @staticmethod
    def filter_id(results, query_id):
        return [res for res in results if res[0] != query_id]

    def search(self, method: SearchType, query_id: str, query_string: str = None, query_embedding: str = None, top_n=5, tokenize_query=True, table_name=["corpus"], rrf_k=60):
        results = []
        if tokenize_query:
            query_string = tokenize(query_string)
        if method == SearchType.SPARSE:
            results = self.fts_search(query_string, top_n=top_n, table_name=table_name[0])
        elif method == SearchType.DENSE:
            results = self.embedding_search(query_embedding, top_n=top_n, table_name=table_name[0])
        elif method == SearchType.HYBRID:
            results = self.rrf_search(query_string, query_embedding, top_n=top_n, k=rrf_k, table_name=table_name)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        return self.filter_id(results, query_id)
    
    @abstractclassmethod
    def get_search_type(self, table_name: str) -> SearchType:
        """
        Returns the type of search this class implements.
        Should return an instance of SearchType.
        """
        pass
    
    @abstractclassmethod
    def fts_search(self, query_string: str, top_n=5, table_name="corpus"):
        pass

    @abstractclassmethod
    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus"):
        pass

    @abstractclassmethod
    def rrf_search(self, query_string: str, query_embedding: str, top_n=5, k=60, table_name=["sparse", "dense"]):
        pass