from enum import Enum

class IndexType(Enum):
    SPARSE = 'sparse'
    DENSE = 'dense'

class SearchType(Enum):
    SPARSE = 'sparse'
    DENSE = 'dense'
    HYBRID = 'hybrid'

class SearchDB(Enum):
    DUCKDB = 'duckdb'
    SQLITE = 'sqlite'
    POSTGRES = 'postgres'