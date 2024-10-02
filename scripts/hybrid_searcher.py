import duckdb
import json
import os
import re
from tqdm import tqdm

class HybridSearcher:
    def __init__(self, db_path=':memory:'):
        self.conn = duckdb.connect(db_path)
        self.embedding_dim = None

    def _create_tables(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS corpus (
                id VARCHAR PRIMARY KEY,
                contents VARCHAR,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS query (
                id VARCHAR PRIMARY KEY,
                contents VARCHAR,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)

    def _insert_embedding(self, table_name, row_data):
        try:
            self.conn.execute(f"""
                INSERT INTO {table_name} (id, contents, embedding)
                VALUES (?, ?, ?)
            """, (row_data['id'], row_data['contents'], row_data['vector']))
        except duckdb.IntegrityError:
            print(f"Skipping duplicate id in {table_name}: {row_data['id']}")

    def _determine_embedding_dim(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    row_data = json.loads(line.strip())
                    return len(row_data['vector'])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        raise ValueError("Could not determine embedding dimension from the file")

    def load_jsonl_to_table(self, file_path, table_name):
        if self.embedding_dim is None:
            self.embedding_dim = self._determine_embedding_dim(file_path)
            self._create_tables()

        with open(file_path, 'r') as file:
            for line in file:
                try:
                    row_data = json.loads(line.strip())
                    self._insert_embedding(table_name, row_data)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in {file_path}: {line}")
        self.conn.commit()

    def initialize_tables(self, corpus_file, query_file):
        self.load_jsonl_to_table(corpus_file, 'corpus')
        self.load_jsonl_to_table(query_file, 'query')
        self.conn.execute("PRAGMA create_fts_index(corpus, id, contents)")
        
        for table in ['corpus', 'query']:
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

    def embedding_search(self, query_id, top_n=5):
        query = f"""
        WITH query_embedding AS (
            SELECT embedding FROM query WHERE id = ?
        )
        SELECT corpus.id, corpus.contents, 
               array_cosine_similarity(corpus.embedding, query_embedding.embedding) AS score
        FROM corpus, query_embedding
        ORDER BY score DESC
        LIMIT ?
        """
        return self.conn.execute(query, [query_id, top_n]).fetchall()

    def convex_combination_search(self, query_id, query_string, alpha=0.8, top_n=5):
        query = f"""
        WITH 
        embd AS (
            SELECT corpus.id, corpus.contents, 
                    array_cosine_similarity(corpus.embedding, q.embedding) AS score
            FROM corpus, (SELECT embedding FROM query WHERE id = ?) AS q
        ),
        fts AS (
            SELECT id, contents, 
                    COALESCE(fts_main_corpus.match_bm25(id, ?), 0) AS score
            FROM corpus
        ),
        normalized_scores as (
            select 
                fts.id, 
                fts.contents, 
                fts.score as raw_score, 
                embd.score as raw_embd_score,
                (fts.score / (select max(score) from fts)) as norm_score,
                ((embd.score + 1) / (select max(score) + 1 from embd)) as norm_embd_score
            from 
                fts
            inner join
                embd 
            on fts.id = embd.id
        )
        SELECT 
            id, 
            contents, 
            --raw_score, 
            --raw_embd_score, 
            --norm_score, 
            --norm_embd_score, 
            ({alpha}*norm_embd_score + {1-alpha}*norm_score) AS score_cc
        FROM normalized_scores
        ORDER BY score_cc DESC
        LIMIT ?;
        """
        return self.conn.execute(query, [query_id, query_string, top_n]).fetchall()

    def rrf(self, query_id, query_string, k=60, top_n=5):
        query = f"""
        WITH 
        embd AS (
            SELECT corpus.id, corpus.contents, 
                array_cosine_similarity(corpus.embedding, q.embedding) AS sim_score,
                ROW_NUMBER() OVER (ORDER BY array_cosine_similarity(corpus.embedding, q.embedding) DESC) AS sim_rank
            FROM corpus, (SELECT embedding FROM query WHERE id = ?) AS q
        ),
        fts AS (
            SELECT id, contents, 
                COALESCE(fts_main_corpus.match_bm25(id, ?), 0) AS fts_score,
                ROW_NUMBER() OVER (ORDER BY COALESCE(fts_main_corpus.match_bm25(id, ?), 0) DESC) AS fts_rank
            FROM corpus
        ),
        combined_results AS (
            SELECT 
                COALESCE(s.id, f.id) AS id,
                COALESCE(s.contents, f.contents) AS contents,
                COALESCE(1.0 / ({k} + s.sim_rank), 0) + COALESCE(1.0 / ({k} + f.fts_rank), 0) AS rrf_score
            FROM embd s
            FULL OUTER JOIN fts f ON s.id = f.id
        )
        SELECT id, contents, rrf_score
        FROM combined_results
        ORDER BY rrf_score DESC
        LIMIT ?
        """
        return self.conn.execute(query, [query_id, query_string, query_string, top_n]).fetchall()

    def _custom_sort_key(self, item):
        # The default sorting in DuckDB is string comparison, which does not put the IDs in numerical strictly increasing order 
        query_id = item[0]
        rank = item[3]
        parts = re.split(r'(\d+)', query_id)
        parts = [int(part) if part.isdigit() else part for part in parts]
        return (parts, rank)

    def search_all_queries(self, search_method, output_file, top_n=1000, run_tag=None, **kwargs):
        queries = self.conn.execute("SELECT id, contents FROM query").fetchall()
        
        run_tag = run_tag or search_method.__name__
        
        all_results = []
        
        for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
            if search_method == self.embedding_search:
                results = search_method(query_id, top_n=top_n, **kwargs)
            elif search_method == self.fts_search:
                results = search_method(query_string, top_n=top_n, **kwargs)
            else:
                results = search_method(query_id, query_string, top_n=top_n, **kwargs)
            
            for rank, (doc_id, _, score) in enumerate(results, 1):
                all_results.append((query_id, doc_id, score, rank))
        
        all_results.sort(key=self._custom_sort_key)
        
        print(f"Writing sorted results to file for {run_tag}...")
        with open(output_file, 'w') as f:
            for query_id, doc_id, score, rank in all_results:
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")
        
        print(f"Completed {run_tag}")

    def run_trec(self, out_dir, run_tags=[None, None, None, None], top_n=1000, rrf_k=60, cc_alpha=0.8):
        os.makedirs(out_dir, exist_ok=True)
        
        search_methods = [self.fts_search, self.embedding_search, self.rrf, self.convex_combination_search]
        
        for method, run_tag in zip(search_methods, run_tags):
            run_tag = run_tag or method.__name__
            
            output_file = os.path.join(out_dir, f"run.{run_tag}.txt")
            
            kwargs = {}
            if method == self.rrf:
                kwargs['k'] = rrf_k
            elif method == self.convex_combination_search:
                kwargs['alpha'] = cc_alpha
            
            self.search_all_queries(
                method,
                output_file,
                run_tag=run_tag,
                top_n=top_n,
                **kwargs
            )

if __name__ == "__main__":
    searcher = HybridSearcher()
    searcher.initialize_tables('../indexes/nfcorpus.bge-base-en-v1.5/corpus_embeddings.jsonl', 
                               '../indexes/nfcorpus.bge-base-en-v1.5/query_embeddings.jsonl')
    searcher.run_trec("../runs/nfcorpus", ["bm25", "bge-base-en-v1.5", "rrf-60", "cc-0.8"])