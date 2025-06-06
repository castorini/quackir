from enum import Enum

class SearchType(Enum):
    FTS = 'fts'
    EMBD = 'embd'
    RRF = 'rrf'

class SearchDB(Enum):
    DUCKDB = 'duckdb'
    SQLITE = 'sqlite'
    POSTGRES = 'postgres'