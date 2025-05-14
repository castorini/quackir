import argparse
from tqdm import tqdm
import sys
import re
from psql_searcher import Postgres_Searcher
from sqlite_searcher import SQLite_Searcher
from duckdb_searcher import DuckDB_Searcher

def _custom_sort_key(item):
    # The default sorting in DuckDB is string comparison, which does not put the IDs in numerical strictly increasing order 
    query_id = item[0]
    rank = item[3]
    parts = re.split(r'(\d+)', query_id)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return (parts, rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", type=str, required=True)
    parser.add_argument("--query-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--db", type=str, choices=["sqlite", "duckdb", "postgres"], required=True)
    parser.add_argument("--method", type=str, choices=["fts", "bge", "rrf"], required=True)
    parser.add_argument("--pretokenized", action='store_true', required=False)

    args = parser.parse_args()

    if args.db == 'sqlite' and args.method != 'fts':
        print("Sorry, SQLite search currently only supports the fts method.")
        sys.exit()

    searcher = None
    if args.db == 'postgres':
        searcher = Postgres_Searcher()
    elif args.db == 'sqlite':
        searcher = SQLite_Searcher()
    elif args.db == 'duckdb':
        searcher = DuckDB_Searcher()

    searcher.init_tables("corpus", args.corpus_file, args.method, args.pretokenized)
    searcher.init_tables("query", args.query_file, args.method, args.pretokenized)
    if args.method != 'bge':
        searcher.fts_index()

    queries = searcher.get_queries()
    run_tag = args.method + '_' + args.db

    all_results = []

    for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
        if args.method == 'fts':
            results = searcher.fts_search(query_string, top_n=1000)
        elif args.method == 'bge':
            results = searcher.embedding_search(query_id, top_n=1000)
        elif args.method == 'rrf':
            results = searcher.rrf(query_id, query_string, top_n=1000)
        for rank, (doc_id, score) in enumerate(results, 1):
            all_results.append((query_id, doc_id, score, rank))

    all_results.sort(key=_custom_sort_key) 

    with open(args.output_file, "w") as f:
        for query_id, doc_id, score, rank in all_results:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")