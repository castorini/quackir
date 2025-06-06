from quackir._base import IndexType, _add_db_parser_arguments, _load_env, SearchDB
from ._util import get_indexer
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and index a database.")
    _add_db_parser_arguments(parser)

    parser.add_argument("--input", type=str, required=True, help="Path to the file containing data. Must be in jsonl format with 'id' and either/or 'contents', 'vector' fields.") 
    parser.add_argument("--index", type=str, default="corpus", help="Name of the table to create")
    parser.add_argument("--index-type", type=IndexType, choices=list(IndexType), required=True, help="Type of index to create.")
    parser.add_argument("--pretokenized", action='store_true', default=False, help="Indicate if the contents are pretokenized. Default is False, meaning the contents will be tokenized during indexing.")
    parser.add_argument("--dimension", type=int, default=768, help="Dimension of the embedding vector")

    args = parser.parse_args()
    _load_env(args)
    if args.db_type == SearchDB.SQLITE and args.index_type != IndexType.SPARSE:
        print("Sorry, SQLite indexing currently only supports the sparse method.")
        sys.exit()

    indexer = get_indexer(
        db_type=args.db_type,
        db_path=args.db_path,
        db_name=args.db_name,
        db_user=args.db_user
    )

    indexer.init_table(args.index, args.input, args.index_type, args.pretokenized, args.dimension)
    if args.index_type == IndexType.SPARSE:
        indexer.fts_index(args.index)