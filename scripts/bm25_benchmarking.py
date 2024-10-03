import os
import json
import argparse
import csv
import re
from tqdm import tqdm
import duckdb


class BM25Searcher:
    def __init__(self) -> None:
        self.conn = duckdb.connect(":memory:")

    def _create_tables(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS corpus (
                id VARCHAR PRIMARY KEY,
                contents VARCHAR,
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS query (
                id VARCHAR PRIMARY KEY,
                contents VARCHAR,
            )
        """)

    def _add_to_index(self, id, text, table_name):
        try:
            self.conn.execute(
                f"""
                INSERT INTO {table_name} (id, contents) VALUES (?, ?)
            """,
                (id, text),
            )
        except duckdb.IntegrityError:
            print(f"Skipping duplicate id in {table_name}: {id}")

    def _load_tsv_to_table(self, tsv_file, table_name):
        with open(tsv_file, "r") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")

            # Iterate through each row, assuming no header
            for row in reader:
                id = row[0]
                text = row[1]
                self._add_to_index(id, text, table_name)
        self.conn.commit()

    def _load_jsonl_to_table(self, file_path, table_name):
        with open(file_path, "r") as file:
            for line in file:
                try:
                    row_data = json.loads(line.strip())
                    self._add_to_index(row_data["_id"], row_data["text"], table_name)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}: {line}")
        self.conn.commit()

    def init_tables(self, corpus_file, query_file):
        self._create_tables()
        self._load_jsonl_to_table(corpus_file, "corpus")
        self._load_jsonl_to_table(query_file, "query")

        self.conn.execute("PRAGMA create_fts_index(corpus, id, contents)")

        for table in ["corpus", "query"]:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"Loaded {result.fetchone()[0]} rows in {table}")

    def fts_search(self, query_string, top_n=5):
        query = """
        WITH fts AS (
            SELECT *, COALESCE(fts_main_corpus.match_bm25(id, ?), 0) AS score
            FROM corpus    
        )        
        SELECT id, contents, score
        FROM fts
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?;
        """
        return self.conn.execute(query, [query_string, top_n]).fetchall()

    def _custom_sort_key(self, item):
        # The default sorting in DuckDB is string comparison, which does not put the IDs in numerical strictly increasing order
        query_id = item[0]
        rank = item[3]
        parts = re.split(r"(\d+)", query_id)
        parts = [int(part) if part.isdigit() else part for part in parts]
        return (parts, rank)

    def search_all_queries(self, output_file, top_n=1000, run_tag=None, **kwargs):
        queries = self.conn.execute("SELECT id, contents FROM query").fetchall()

        run_tag = run_tag or "bm25_run"

        all_results = []

        for query_id, query_string in tqdm(
            queries, desc=f"Processing {run_tag}", unit="query"
        ):
            results = self.fts_search(query_string, top_n=top_n)

            for rank, (doc_id, _, score) in enumerate(results, 1):
                all_results.append((query_id, doc_id, score, rank))

        all_results.sort(key=self._custom_sort_key)

        print(f"Writing sorted results to file for {run_tag}...")
        with open(output_file, "w") as f:
            for query_id, doc_id, score, rank in all_results:
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")

        print(f"Completed {run_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", type=str)
    parser.add_argument("--query-file", type=str)

    args = parser.parse_args()

    benchmarker = BM25Searcher()
    benchmarker.init_tables(args.corpus_file, args.query_file)
    # results = benchmarker.conn.execute("SELECT * FROM corpus LIMIT 5;").fetchall()
    #
    benchmarker.search_all_queries("results.txt")
