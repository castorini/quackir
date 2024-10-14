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

    # def _create_tables(self):
    #     self.conn.execute(f"""
    #         CREATE TABLE IF NOT EXISTS corpus (
    #             id VARCHAR PRIMARY KEY,
    #             contents VARCHAR,
    #         )
    #     """)
    #     self.conn.execute(f"""
    #         CREATE TABLE IF NOT EXISTS query (
    #             id VARCHAR PRIMARY KEY,
    #             contents VARCHAR,
    #         )
    #     """)

    def _add_to_index(self, id, text, table_name):
        try:
            self.conn.execute(
                f"""
                INSERT INTO {table_name} (id, contents) VALUES (?, ?)
            """,
                (id, text),
            )
            print("inserted", id, text, "into", table_name)
        except duckdb.IntegrityError:
            print(f"Skipping duplicate id in {table_name}: {id}")

    def batch_load_tsv_to_table(self, tsv_file, table_name):
        print("Loading: ", tsv_file)
        self.conn.execute(f"""
                        CREATE TABLE {table_name} AS
                        SELECT column0 as id, column1 as contents FROM read_csv_auto('{tsv_file}', delim='\t', header=False);
                    """)

    def _load_jsonl_to_table(self, file_path, table_name):
        with open(file_path, "r") as file:
            for line in file:
                try:
                    row_data = json.loads(line.strip())
                    self._add_to_index(row_data["_id"], row_data["text"], table_name)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}: {line}")
        self.conn.commit()

    def batch_load_jsonl_to_table(self, file_path, table_name):
        print("Loading: ", file_path)
        self.conn.execute(f"""
            CREATE TABLE {table_name} AS SELECT _id AS id, text AS contents FROM read_json('{file_path}', format = 'newline_delimited');
        """)

    def init_tables(self, corpus_file, query_file):
        self.batch_load_tsv_to_table(query_file, "query")
        self.batch_load_jsonl_to_table(corpus_file, "corpus")

        self.conn.execute("PRAGMA create_fts_index(corpus, id, contents)")

        for table in ["corpus", "query"]:
            # for table in ["query"]:
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

        run_tag = run_tag or "bm25_duckDB"

        all_results = []

        for query_id, query_string in tqdm(
            queries, desc=f"Processing {run_tag}", unit="query"
        ):
            results = self.fts_search(query_string, top_n=top_n)

            for rank, (doc_id, _, score) in enumerate(results, 1):
                all_results.append((query_id, doc_id, score, rank))
                print((query_id, doc_id, score, rank))

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
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    benchmarker = BM25Searcher()
    benchmarker.init_tables(args.corpus_file, args.query_file)
    # results = benchmarker.conn.execute("SELECT * FROM corpus LIMIT 5;").fetchall()
    #
    benchmarker.search_all_queries(args.output_file)
