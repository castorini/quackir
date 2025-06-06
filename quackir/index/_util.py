from ._duck import DuckDBIndexer
from ._sqlite import SQLiteIndexer
from ._postgres import PostgresIndexer
from quackir.common.enums import SearchDB

def get_indexer(db_type: SearchDB, db_path: str = "database.db", db_name: str = "quackir", user: str = "postgres") -> object:
    """
    Factory function to get the appropriate indexer based on the database type.
    
    Args:
        db_type (SearchDB): The type of database to use.
        db_path (str): Path to the database file for DuckDB and SQLite. Ignored for Postgres.
        db_name (str): Name of the database for Postgres. Ignored for DuckDB and SQLite.
        user (str): Username for Postgres. Ignored for DuckDB and SQLite.
    
    Returns:
        object: An instance of an indexer class corresponding to the specified database type.
    """
    if db_type == SearchDB.DUCKDB:
        return DuckDBIndexer(db_path)
    elif db_type == SearchDB.SQLITE:
        return SQLiteIndexer(db_path)
    elif db_type == SearchDB.POSTGRES:
        return PostgresIndexer(db_name, user)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")