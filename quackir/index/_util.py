#
# QuackIR: Reproducible IR research in RDBMS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ._duck import DuckDBIndexer
from ._sqlite import SQLiteIndexer
from ._postgres import PostgresIndexer
from quackir._base import SearchDB

def get_indexer(db_type: SearchDB, db_path: str = "database.db", db_name: str = "quackir", db_user: str = "postgres") -> object:
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
        return PostgresIndexer(db_name, db_user)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")