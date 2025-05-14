import duckdb
import json
from db_searcher import DB_Searcher

class DuckDB_Searcher(DB_Searcher):
    def __init__(self):
        self.conn = duckdb.connect("duck.db")

    def init_tables(self, table_name, file_path, method):
        self.conn.execute(f"drop table if exists {table_name}")
        if method == 'fts':
            self.conn.execute(f"""CREATE TABLE {table_name} (id varchar primary key, contents varchar)""")
        else:
            self.conn.execute(f"""create table {table_name} (id varchar primary key, contents varchar, embedding double[768])""")
            
        with open(file_path, 'r') as file:
            for line in file:
                row = json.loads(line.strip())
                if method == 'fts':
                    self.conn.execute(f"insert into {table_name} (id, contents) values (?, ?)", (row['id'], row['contents']))
                else:
                    self.conn.execute(f"""insert into {table_name} (id, contents, embedding) values (?, ?, ?)""", (row['id'], row['contents'], row['vector']))
        self.conn.execute(f"select count(*) from {table_name}")
        print(f"Loaded {self.conn.fetchone()[0]} rows in {table_name}")

    def fts_search(self, query_string, top_n=5):
        query = """
        WITH fts AS (
            SELECT *, COALESCE(fts_main_corpus.match_bm25(id, ?, k:=0.9, b:=0.4), 0) AS score
            FROM corpus
        )
        SELECT id, score
        FROM fts
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?;
        """
        return self.conn.execute(query, [query_string, top_n]).fetchall()

    def fts_index(self):
        self.conn.execute("PRAGMA create_fts_index(corpus, id, contents, stemmer = 'none', stopwords = 'none', ignore = 'a^', strip_accents = 0, lower = 0, overwrite = 1)")

    def embedding_search(self, query_id, top_n=5):
        query = f"""
        WITH query_embedding AS (
            SELECT embedding FROM query WHERE id = ?
        )
        SELECT corpus.id, 
                array_cosine_similarity(corpus.embedding, query_embedding.embedding) AS score
        FROM corpus, query_embedding
        ORDER BY score DESC
        LIMIT ?
        """
        return self.conn.execute(query, [query_id, top_n]).fetchall()

    def rrf(self, query_id, query_string, k=60, top_n=5):
        #         ORDER BY array_cosine_similarity(corpus.embedding, query_embedding.embedding) desc
        # ORDER BY COALESCE(fts_main_corpus.match_bm25(id, ?), 0) DESC
        query = f"""
        WITH 
        embd AS (
            SELECT corpus.id,
                ROW_NUMBER() OVER (ORDER BY array_cosine_similarity(corpus.embedding, q.embedding) DESC) AS sim_rank
            FROM corpus, (SELECT embedding FROM query WHERE id = ?) AS q
            limit 1000
        ),
        fts AS (
            SELECT id, 
                ROW_NUMBER() OVER (ORDER BY COALESCE(fts_main_corpus.match_bm25(id, ?), 0) DESC) AS fts_rank
            FROM corpus
            limit 1000
        ),
        combined_results AS (
            SELECT 
                COALESCE(s.id, f.id) AS id,
                COALESCE(1.0 / ({k} + s.sim_rank), 0) + COALESCE(1.0 / ({k} + f.fts_rank), 0) AS rrf_score
            FROM embd s
            FULL OUTER JOIN fts f ON s.id = f.id
        )
        SELECT id, rrf_score
        FROM combined_results
        ORDER BY rrf_score DESC
        LIMIT ?
        """
        return self.conn.execute(query, [query_id, query_string, top_n]).fetchall()
    
    def get_queries(self):
        return self.conn.execute("SELECT id, contents FROM query").fetchall()