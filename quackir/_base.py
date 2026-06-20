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

from enum import Enum
import dotenv
import argparse
import os

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

class SecretProvider(Enum):
    OLLAMA = 'OLLAMA'
    OPENAI = 'OPENAI'
    AZURE = 'AZURE_LLM'

def count_lines(filename, open_cmd):
    with open_cmd(filename, 'r') as file:
        return sum(1 for _ in file)

def sanitize_table_name(table_name: str):
    return table_name.replace("-", "_")

def _add_db_parser_arguments(parser: argparse.ArgumentParser):
    """
    Adds common arguments to the provided parser for database and search configurations.
    
    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument("--db-type", type=SearchDB, choices=list(SearchDB), help="Type of database to use.")
    parser.add_argument("--db-path", type=str, default="database.db", help="Path to the database file used for DuckDB and SQLite. Ignored for Postgres.")
    parser.add_argument("--db-name", type=str, default="quackir", help="Name of the database for Postgres. Ignored for DuckDB and SQLite.")
    parser.add_argument("--db-user", type=str, default="postgres", help="Username for Postgres. Ignored for DuckDB and SQLite.")

def _load_env(args):
    """
    Load environment variables from a .env file if it exists.
    This is useful for setting up database connection parameters.
    """
    dotenv.load_dotenv()
    args.db_type = os.getenv('DB_TYPE', args.db_type)
    args.db_path = os.getenv('DB_PATH', args.db_path)
    args.db_name = os.getenv('DB_NAME', args.db_name)
    args.db_user = os.getenv('DB_USER', args.db_user)
    if not args.db_type:
        raise ValueError("Database type must be specified using --db-type or DB_TYPE environment variable.")
    if args.db_type == SearchDB.POSTGRES:
        if not args.db_user:
            raise ValueError("Postgres requires a user to be specified.")
        if not args.db_name:
            raise ValueError("Postgres requires a database name to be specified.")
    elif args.db_type in [SearchDB.DUCKDB, SearchDB.SQLITE]:
        if not args.db_path:
            raise ValueError(f"{args.db_type.value} requires a database path to be specified.")