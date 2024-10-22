import os
import json
import argparse
import csv
import re
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
from jsonl_to_tsv import convert_jsonl_to_tsv

class PostgreSQLBM25Searcher:
    def __init__(self, db_name, user, password, host='localhost', port=5432, table_prefix=''):
        self.conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.conn.cursor()
        self.table_prefix = self._sanitize_table_name(table_prefix)

    def _sanitize_table_name(self, name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _create_tables(self):
        self.cursor.execute(f'DROP TABLE IF EXISTS "{self.table_prefix}_corpus"')
        self.cursor.execute(f'DROP TABLE IF EXISTS "{self.table_prefix}_query"')
        
        self.cursor.execute(f'''
            CREATE TABLE "{self.table_prefix}_corpus" (
                id VARCHAR PRIMARY KEY,
                contents TEXT
            )
        ''')
        self.cursor.execute(f'''
            CREATE TABLE "{self.table_prefix}_query" (
                id VARCHAR PRIMARY KEY,
                contents TEXT
            )
        ''')
        self.conn.commit()

    def _batch_load_tsv(self, file_path, table_name):
        try:
            with open(file_path, 'r') as f:
                has_header = csv.Sniffer().has_header(f.read(1024))
                f.seek(0)

            with self.conn.cursor() as cur:
                if has_header:
                    cur.copy_expert(f'COPY "{table_name}" (id, contents) FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\t\', HEADER)', open(file_path, 'r'))
                else:
                    cur.copy_expert(f'COPY "{table_name}" (id, contents) FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\t\')', open(file_path, 'r'))
            self.conn.commit()
            print(f"Successfully loaded {file_path} into {table_name}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error loading {file_path} into {table_name}: {str(e)}")

    def _detect_file_format(self, file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            try:
                json.loads(first_line)
                return 'jsonl'
            except json.JSONDecodeError:
                if '\t' in first_line:
                    return 'tsv'
                else:
                    raise ValueError(f"Unable to detect file format for {file_path}. Please ensure it's either JSONL or TSV.")

    def init_tables(self, corpus_file, query_file):
        self._create_tables()
        
        corpus_format = self._detect_file_format(corpus_file)
        query_format = self._detect_file_format(query_file)
        
        print(f"Detected corpus file format: {corpus_format}")
        print(f"Detected query file format: {query_format}")
        
        if corpus_format == 'jsonl':
            corpus_tsv = convert_jsonl_to_tsv(corpus_file)
            self._batch_load_tsv(corpus_tsv, f"{self.table_prefix}_corpus")
        else:
            self._batch_load_tsv(corpus_file, f"{self.table_prefix}_corpus")
        
        if query_format == 'jsonl':
            query_tsv = convert_jsonl_to_tsv(query_file)
            self._batch_load_tsv(query_tsv, f"{self.table_prefix}_query")
        else:
            self._batch_load_tsv(query_file, f"{self.table_prefix}_query")

        self.cursor.execute(f'''
            CREATE INDEX IF NOT EXISTS "{self.table_prefix}_corpus_contents_gin" ON "{self.table_prefix}_corpus" USING gin(to_tsvector('english', contents));
        ''')
        self.conn.commit()

        for table in [f"{self.table_prefix}_corpus", f"{self.table_prefix}_query"]:
            self.cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            result = self.cursor.fetchone()
            print(f"Loaded {result[0]} rows in {table}")

    def _clean_query(self, query_string):
        cleaned = re.sub(r'[^\w\s]', ' ', query_string)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def fts_search(self, query_string, top_n=1000):
        cleaned_query = self._clean_query(query_string)
        ts_query = " | ".join(cleaned_query.split())
        query = f"""
        SELECT 
            id, 
            contents,
            ts_rank(to_tsvector('english', contents), to_tsquery('english', %s), 1|32) AS score
        FROM 
            {self.table_prefix}_corpus
        WHERE
            to_tsvector('english', contents) @@ to_tsquery('english', %s)
        ORDER BY 
            score DESC
        LIMIT %s
        """
        self.cursor.execute(query, (ts_query, ts_query, top_n))
        return self.cursor.fetchall()

    def search_all_queries(self, output_file, top_n=1000, run_tag=None, log_statistics=False):
        self.cursor.execute(f"SELECT id, contents FROM {self.table_prefix}_query")
        queries = self.cursor.fetchall()
        run_tag = run_tag or f"postgresql_fts_{self.table_prefix}_run"
        all_results = []
        query_result_counts = {}

        with tqdm(queries, desc=f"Processing {run_tag}", unit="query") as pbar:
            for query_id, query_string in pbar:
                results = self.fts_search(query_string, top_n=top_n)
                query_result_counts[query_id] = len(results)
                for rank, (doc_id, _, score) in enumerate(results, 1):
                    all_results.append((query_id, doc_id, score, rank))
                if log_statistics:
                    pbar.set_postfix(results=len(results))

        all_results.sort(key=self._custom_sort_key)

        if log_statistics:
            print(f"\nWriting sorted results to file for {run_tag}...")
        
        with open(output_file, "w") as f:
            for query_id, doc_id, score, rank in all_results:
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")

        if log_statistics:
            print(f"Completed {run_tag}")
            print(f"Total results written: {len(all_results)}")
            print(f"Number of queries with results: {len([c for c in query_result_counts.values() if c > 0])}")
            print(f"Number of queries with no results: {len([c for c in query_result_counts.values() if c == 0])}")
            print(f"Average results per query: {sum(query_result_counts.values()) / len(query_result_counts):.2f}")
            print(f"Min results for a query: {min(query_result_counts.values())}")
            print(f"Max results for a query: {max(query_result_counts.values())}")

            stats_file = f"{output_file}_query_stats.tsv"
            with open(stats_file, "w") as f:
                f.write("query_id\tresult_count\n")
                for query_id, count in query_result_counts.items():
                    f.write(f"{query_id}\t{count}\n")
            print(f"Detailed query statistics written to {stats_file}")

    def _custom_sort_key(self, item):
        query_id = item[0]
        rank = item[3]
        parts = re.split(r"(\d+)", query_id)
        parts = [int(part) if part.isdigit() else part for part in parts]
        return (parts, rank)

    def close(self):
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--db-name", type=str, required=True)
    parser.add_argument("--db-user", type=str, required=True)
    parser.add_argument("--db-password", type=str, required=True)
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=int, default=5433)
    parser.add_argument("--table-prefix", type=str, required=True)
    args = parser.parse_args()

    benchmarker = PostgreSQLBM25Searcher(
        db_name=args.db_name,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
        table_prefix=args.table_prefix
    )
    print(f"Processing {args.corpus}")
    benchmarker.init_tables(args.corpus, args.query)
    benchmarker.search_all_queries(args.output, log_statistics=True)
    benchmarker.close()