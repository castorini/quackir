import psycopg2
import json
import re
from db_searcher import DB_Searcher

class Postgres_Searcher(DB_Searcher):
    def __init__(self):
        self.conn = psycopg2.connect(dbname='beir_datasets', user='postgres')

    def init_tables(self, table_name, file_path, method, pretokenized=False):
        cur = self.conn.cursor()
        cur.execute(f"drop table if exists {table_name}")

        if method == 'fts':
            cur.execute(f"create table {table_name} (id text, contents text);")
        else:
            cur.execute(f"create table {table_name} (id text, contents text, embedding vector(768));")
        
        with open(file_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                if not pretokenized:
                    contents = self.tokenize(row['contents'])
                else:
                    contents = row['contents']
                if method == 'fts':
                    cur.execute(f"insert into {table_name} (id, contents) values (%s, %s)", (row['id'], contents))
                else:
                    cur.execute(f"insert into {table_name} (id, contents, embedding) values (%s, %s, %s)", (row['id'], contents, row['vector']))
        
        self.conn.commit()
        cur.execute(f"select count(*) from {table_name}")
        print(f"Loaded {cur.fetchone()[0]} rows in {table_name}")

    def clean_tsquery(self, query_string):
        cleaned_query = re.sub(r'[^\w\s]', ' ', query_string)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        ts_query = " | ".join(cleaned_query.split())
        return ts_query
    
    def fts_search(self, query_string, top_n=5, tokenize_query=False):
        if tokenize_query:
            query_string = self.tokenize(query_string)
        ts_query = self.clean_tsquery(query_string)
        query = f"""
        SELECT 
            id, 
            ts_rank(to_tsvector('simple', contents), to_tsquery('simple', %s)) AS score
        FROM 
            corpus
        WHERE
            to_tsvector('simple', contents) @@ to_tsquery('simple', %s)
        ORDER BY 
            score DESC
        LIMIT %s
        """
        cur = self.conn.cursor()
        cur.execute(query, (ts_query, ts_query, top_n))
        return cur.fetchall()
    
    def fts_index(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE INDEX "corpus_contents_gin" ON "corpus" USING gin(to_tsvector('simple', contents));''')

    def embedding_search(self, query_id, top_n=5):
        cur = self.conn.cursor()
        cur.execute("select embedding from query where id = %s", (query_id,))
        vector = cur.fetchone()[0]
        query = """select id, 1 - (embedding <=> %s::vector) as score from corpus order by score desc limit %s"""
        cur.execute(query, (vector, top_n))
        return cur.fetchall()
    
    def rrf(self, query_id: str, query_string: str, top_n=5, k=60):
        ts_query = self.clean_tsquery(query_string)
        cur = self.conn.cursor()
        cur.execute("select embedding from query where id = %s", (query_id,))
        vector = cur.fetchone()[0]
        sql = """
        WITH semantic_search AS (
            SELECT id, RANK () OVER (ORDER BY embedding <=> %(vector)s) AS rank
            FROM corpus
            LIMIT %(n)s
        ),
        keyword_search AS (
            SELECT id, RANK () OVER (ORDER BY ts_rank(to_tsvector('simple', contents), query) DESC) as rank
            FROM corpus, to_tsquery('simple', %(query)s) query
            WHERE to_tsvector('simple', contents) @@ query
            LIMIT %(n)s
        )
        SELECT
            COALESCE(semantic_search.id, keyword_search.id) AS id,
            COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
        FROM semantic_search
        FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
        ORDER BY score DESC
        LIMIT %(n)s
        """
        cur.execute(sql, {'query': ts_query, 'vector': vector, 'n': top_n, 'k': k})
        results = cur.fetchall()
        return results

    def get_queries(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, contents FROM query")
        return cur.fetchall()