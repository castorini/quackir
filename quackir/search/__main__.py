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

from quackir._base import SearchType, SearchDB, _add_db_parser_arguments, _load_env, sanitize_table_name
from ._util import get_searcher, _custom_sort_key
import argparse
import json
import sys
import gzip
from tqdm import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for queries in a database.")
    _add_db_parser_arguments(parser)

    parser.add_argument("--topics", type=str, required=True, help="Path to the file containing queries in jsonl format with the fields id, contents/vector.")
    parser.add_argument("--search-method", type=SearchType, choices=list(SearchType), help="Method of search to perform.")
    parser.add_argument("--index", type=str, default=["corpus"], nargs='+', help="Name of the table to search in. Accepts two values for hybrid search, one sparse and one dense; one value for sparse or dense search.")
    parser.add_argument("--pretokenized", action='store_true', default=False, help="Indicate if the queries are pretokenized. Default is False, meaning the queries will be tokenized during search.")
    parser.add_argument("--hits", type=int, default=1000, help="Number of top results to return")
    parser.add_argument("--rrf-k", type=int, default=60, help="Parameter k needed for reciprocal rank fusion. Ignored for other search methods.")

    parser.add_argument("--output", type=str, required=True, help="Path to save the search results") 
    parser.add_argument("--run-tag", type=str, help="Tag to identify the run in the output file")

    args = parser.parse_args()
    _load_env(args)

    if args.db_type == SearchDB.SQLITE and args.search_method != None and args.search_method != SearchType.SPARSE:
        print("Sorry, SQLite search currently only supports the sparse method.")
        sys.exit()
    if len(args.index) > 2:
        raise ValueError("Invalid number of table names provided. Must be 1 or 2.")
    if len(args.index) == 2 and args.search_method != None and args.search_method != SearchType.HYBRID:
        raise ValueError("If two table names are provided, the search method must be Hybrid (Reciprocal Rank Fusion).")
    if args.search_method == SearchType.HYBRID and len(args.index) != 2:
        raise ValueError("Hybrid search requires exactly two table names, one for sparse and one for dense search.")

    searcher = get_searcher(
        db_type=args.db_type,
        db_path=args.db_path,
        db_name=args.db_name,
        db_user=args.db_user
    )

    args.index = [sanitize_table_name(index) for index in args.index]

    if not args.search_method:
        if len(args.index) == 1:
            args.search_method = searcher.get_search_type(table_name=args.index[0])
        elif len(args.index) == 2:
            searcher_type1 = searcher.get_search_type(table_name=args.index[0])
            searcher_type2 = searcher.get_search_type(table_name=args.index[1])
            if searcher_type1 != searcher_type2:
                args.search_method = SearchType.HYBRID
            else:
                raise ValueError("If two table names are provided, they must be of different types (sparse and dense) for hybrid search.")

    if not args.run_tag:
        args.run_tag = f"{args.search_method.value}_{args.db_type.value}"

    queries = []
    open_cmd = open
    if args.topics.endswith('.gz'):
        open_cmd = gzip.open
    with open_cmd(args.topics, 'rt') as f:
        for line in f:
            if '.jsonl' in args.topics:
                query = json.loads(line.strip())
            elif '.tsv' in args.topics:
                parts = line.strip().split('\t')
                query = {"id": parts[0], "contents": parts[1]}
            queries.append(query)
    print(f"Loaded {len(queries)} queries from {args.topics}")

    all_results = []
    all_times = []
    for query in tqdm(queries, desc=f"Processing {args.run_tag}", unit="query", total=len(queries)):
        query_id = query.get("id", query.get("qid", None))
        start_time = time.time()
        results = searcher.search(
            method=args.search_method,
            query_id=query_id,
            query_string=query.get("contents", None),
            query_embedding=query.get("vector", None),
            top_n=args.hits,
            tokenize_query=not args.pretokenized,
            table_names=args.index,
            rrf_k=args.rrf_k
        )
        end_time = time.time()
        all_times.append(end_time - start_time)
        for rank, (doc_id, score) in enumerate(results, 1):
            all_results.append((query_id, doc_id, score, rank))

    print(f"Processed at {len(queries) / sum(all_times)} queries per second.")
    all_results.sort(key=_custom_sort_key) 

    with open(args.output, "w") as f:
        for query_id, doc_id, score, rank in all_results:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {args.run_tag}\n")
    print(f"Search results saved to {args.output}")