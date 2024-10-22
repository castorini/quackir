import os
import json
import argparse
import csv
import re
import sqlite3
import duckdb
import gzip
from tqdm import tqdm


class BaseBM25Searcher:
    def __init__(self, db_type) -> None:
        self.conn = None
        self.db_type = db_type

    def _add_to_index(self, id, text, table_name):
        try:
            self.conn.execute(
                f"""
                INSERT INTO {table_name} (id, contents) VALUES (?, ?)
            """,
                (id, text),
            )
            # print("inserted", id, text, "into", table_name)
        except Exception as e:
            print(f"Skipping duplicate id in {table_name}: {id} - {e}")

    def _custom_sort_key(self, item):
        # The default sorting in SQLite and DuckDB is string comparison, which does not put the IDs in numerical strictly increasing order
        query_id = item[0]
        rank = item[3]
        parts = re.split(r"(\d+)", query_id)
        parts = [int(part) if part.isdigit() else part for part in parts]
        return (parts, rank)

    def search_all_queries(self, output_file, top_n=1000, run_tag=None, **kwargs):
        queries = self.conn.execute("SELECT id, contents FROM query").fetchall()
        print("Top N: ", top_n)
        run_tag = run_tag or f"bm25_{self.db_type}"

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


class SQLiteBM25Searcher(BaseBM25Searcher):
    def __init__(self) -> None:
        super().__init__("sqlite")
        self.conn = sqlite3.connect(":memory:")
        # self.conn.execute("PRAGMA foreign_keys = 1")

    def batch_load_tsv_to_table(self, tsv_file, table_name):
        print("Loading: ", tsv_file)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        """)
        open_func = gzip.open if tsv_file.endswith(".gz") else open
        with open_func(tsv_file, "rt", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                self._add_to_index(row[0], row[1], table_name)

    def batch_load_jsonl_to_table(self, file_path, table_name):
        print("Loading: ", file_path)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                contents TEXT
            )
        """)
        open_func = gzip.open if file_path.endswith(".gz") else open
        with open_func(file_path, "rt", encoding="utf-8") as file:
            for line in file:
                try:
                    row_data = json.loads(line.strip())
                    self._add_to_index(row_data["_id"], row_data["text"], table_name)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}: {line}")

    def init_tables(self, corpus_file, query_file):
        self.batch_load_tsv_to_table(query_file, "query")
        self.batch_load_jsonl_to_table(corpus_file, "corpus")

        self.conn.execute("""
            CREATE VIRTUAL TABLE fts_corpus USING fts5(
                id, contents,
                content='corpus', content_rowid='rowid'
            )
        """)

        self.conn.execute(
            "INSERT INTO fts_corpus (id, contents) SELECT id, contents FROM corpus;"
        )

        # insert statement needs to be committed
        self.conn.commit()

        for table in ["corpus", "query"]:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"Loaded {result.fetchone()[0]} rows in {table}")

    def prepare_fts5_query(self, query_string, partial=False):
        """Escape the quotes and any special characters, use partial mode for allowing partial matches"""
        if partial:
            # escape single and double quotes
            query_string = query_string.replace("'", "''").replace('"', '""')
            terms = query_string.split()
            escaped_terms = terms
            # make each term a string so that any special chars are escaped
            escaped_terms = [f'"{term}"' for term in escaped_terms]
            # allow matching of any of the terms, using + or AND will turn it into boolean AND retrieval
            res = " OR ".join(escaped_terms)
            return res
        else:
            escaped_query = query_string.replace('"', '""')
            escaped_query = f'"{escaped_query}"'
            return escaped_query

    def fts_search(self, query_string, top_n=5):
        query = """
        SELECT id, contents, bm25(fts_corpus)*-1 AS score
        FROM fts_corpus
        WHERE fts_corpus MATCH '{}'
        ORDER BY score DESC
        LIMIT {}
        """
        prepared_query = self.prepare_fts5_query(query_string, partial=True)
        # print(prepared_query)
        query = query.format(prepared_query, top_n)
        # print(query)
        return self.conn.execute(query).fetchall()


class DuckDBBM25Searcher(BaseBM25Searcher):
    def __init__(self) -> None:
        super().__init__("duckdb")
        self.conn = duckdb.connect(":memory:")

    def batch_load_tsv_to_table(self, tsv_file, table_name):
        print("Loading: ", tsv_file)
        self.conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT column0 as id, column1 as contents FROM read_csv_auto('{tsv_file}', delim='\t', header=False);
        """)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", type=str)
    parser.add_argument("--query-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument(
        "--db-type", type=str, choices=["sqlite", "duckdb"], default="sqlite"
    )
    args = parser.parse_args()

    if args.db_type == "sqlite":
        benchmarker = SQLiteBM25Searcher()
    elif args.db_type == "duckdb":
        benchmarker = DuckDBBM25Searcher()
    else:
        raise ValueError("Unsupported database type. Please use 'sqlite' or 'duckdb'.")

    benchmarker.init_tables(args.corpus_file, args.query_file)
    benchmarker.search_all_queries(args.output_file)
