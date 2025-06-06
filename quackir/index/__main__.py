from quackir.common.enums import SearchType, SearchDB
from ._util import get_indexer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and index a database.")
    parser.add_argument("--db-type", type=SearchDB, choices=list(SearchDB), required=True, help="Type of database to use.")
    parser.add_argument("--db-path", type=str, default="database.db", help="Path to the database file used for DuckDB and SQLite. Ignored for Postgres.")
    parser.add_argument("--db-name", type=str, default="quackir", help="Name of the database for Postgres. Ignored for DuckDB and SQLite.")
    parser.add_argument("--user", type=str, default="postgres", help="Username for Postgres. Ignored for DuckDB and SQLite.")

    parser.add_argument("--table-name", type=str, default="corpus", help="Name of the table to create")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the file containing data. Must be in jsonl format with 'id' and either/or 'contents', 'vector' fields.") 
    parser.add_argument("--index-type", type=SearchType, choices=list(SearchType), required=True, help="Type of index to create.")
    parser.add_argument("--pretokenized", action='store_true', default=False, help="Indicate if the contents are pretokenized. Default is False, meaning the contents will be tokenized during indexing.")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Dimension of the embedding vector")

    args = parser.parse_args()

    indexer = get_indexer(
        db_type=args.db_type,
        db_path=args.db_path,
        db_name=args.db_name,
        user=args.user
    )

    indexer.init_table(args.table_name, args.file_path, args.index_type, args.pretokenized, args.embedding_dim)
    if args.index_type == SearchType.FTS:
        indexer.fts_index(args.table_name)