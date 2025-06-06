from ._duck import DuckDBSearcher
from ._sqlite import SQLiteSearcher
from ._postgres import PostgresSearcher
from quackir._base import SearchDB
import re

def get_searcher(db_type: SearchDB, db_path: str = "database.db", db_name: str = "quackir", user: str = "postgres") -> object:
    """
    Factory function to get the appropriate searcher based on the database type.
    
    Args:
        db_type (SearchDB): The type of database to use.
        db_path (str): Path to the database file for DuckDB and SQLite. Ignored for Postgres.
        db_name (str): Name of the database for Postgres. Ignored for DuckDB and SQLite.
        user (str): Username for Postgres. Ignored for DuckDB and SQLite.
    
    Returns:
        object: An instance of a searcher class corresponding to the specified database type.
    """
    if db_type == SearchDB.DUCKDB:
        return DuckDBSearcher(db_path)
    elif db_type == SearchDB.SQLITE:
        return SQLiteSearcher(db_path)
    elif db_type == SearchDB.POSTGRES:
        return PostgresSearcher(db_name, user)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def _custom_sort_key(item):
    # The default sorting in DuckDB is string comparison, which does not put the IDs in numerical strictly increasing order 
    query_id = item[0]
    rank = item[3]
    parts = re.split(r'(\d+)', query_id)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return (parts, rank)