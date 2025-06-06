from quackir.common.enums import SearchType, SearchDB
from ._util import get_searcher, _custom_sort_key
import argparse
import json
import sys
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for queries in a database.")
    parser.add_argument("--db-type", type=SearchDB, choices=list(SearchDB), required=True, help="Type of database to use.")
    parser.add_argument("--db-path", type=str, default="database.db", help="Path to the database file used for DuckDB and SQLite. Ignored for Postgres.")
    parser.add_argument("--db-name", type=str, default="quackir", help="Name of the database for Postgres. Ignored for DuckDB and SQLite.")
    parser.add_argument("--user", type=str, default="postgres", help="Username for Postgres. Ignored for DuckDB and SQLite.")

    parser.add_argument("--query-file", type=str, required=True, help="Path to the file containing queries in jsonl format with the fields id, contents/vector.")
    parser.add_argument("--table-name", type=str, default="corpus", help="Name of the table to search in")
    parser.add_argument("--search-method", type=SearchType, choices=list(SearchType), required=True, help="Method of search to perform.")
    parser.add_argument("--pretokenized", action='store_true', default=False, help="Indicate if the queries are pretokenized. Default is False, meaning the queries will be tokenized during search.")
    parser.add_argument("--top-k", type=int, default=1000, help="Number of top results to return")
    parser.add_argument("--rrf-k", type=int, default=60, help="Parameter k needed for reciprocal rank fusion. Ignored for other search methods.")

    parser.add_argument("--output-path", type=str, required=True, help="Path to save the search results") 
    parser.add_argument("--run-tag", type=str, help="Tag to identify the run in the output file")

    args = parser.parse_args()

    if args.db_type == SearchDB.SQLITE and args.search_method != SearchType.FTS:
        print("Sorry, SQLite search currently only supports the fts method.")
        sys.exit()

    searcher = get_searcher(
        db_type=args.db_type,
        db_path=args.db_path,
        db_name=args.db_name,
        user=args.user
    )
    
    if not args.run_tag:
        args.run_tag = f"{args.search_method.value}_{args.db_type.value}"

    queries = []
    with open(args.query_file, 'r') as f:
        for line in f:
            query = json.loads(line.strip())
            queries.append(query)
    print(f"Loaded {len(queries)} queries from {args.query_file}")

    all_results = []
    for query in tqdm(queries, desc=f"Processing {args.run_tag}", unit="query", total=len(queries)):
        results = searcher.search(
            method=args.search_method,
            query_id=query["id"],
            query_string=query.get("contents", None),
            query_embedding=query.get("vector", None),
            top_n=args.top_k,
            tokenize_query=not args.pretokenized,
            table_name=args.table_name,
            rrf_k=args.rrf_k
        )
        for rank, (doc_id, score) in enumerate(results, 1):
            all_results.append((query["id"], doc_id, score, rank))

    all_results.sort(key=_custom_sort_key) 

    with open(args.output_path, "w") as f:
        for query_id, doc_id, score, rank in all_results:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {args.run_tag}\n")
    print(f"Search results saved to {args.output_path}")