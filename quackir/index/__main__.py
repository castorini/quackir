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

from quackir._base import IndexType, _add_db_parser_arguments, _load_env, SearchDB, sanitize_table_name
from ._util import get_indexer
import sys
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and index a database.")
    _add_db_parser_arguments(parser)

    parser.add_argument("--input", type=str, required=True, help="Path to the file or folder containing data to index.") 
    parser.add_argument("--index-type", type=IndexType, choices=list(IndexType), required=True, help="Type of index to create.")
    parser.add_argument("--index", type=str, default="corpus", help="Name of the table to create")
    parser.add_argument("--pretokenized", action='store_true', default=False, help="Indicates if the contents are pretokenized. Default is False, meaning the contents will be tokenized during indexing.")
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

    args.index = sanitize_table_name(args.index)
    indexer.init_table(args.index, args.index_type, args.dimension)
    
    if os.path.isdir(args.input):
        with os.scandir(args.input) as files:
            for file in files:
                if file.is_file():
                    indexer.load_table(args.index, file.path, args.index_type, args.pretokenized)
    else:
        indexer.load_table(args.index, args.input, args.index_type, args.pretokenized)
    
    if args.index_type == IndexType.SPARSE:
        indexer.fts_index(args.index)
        print("Sparse index created.")