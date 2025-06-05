# PostgreSQL Full Text Search Baselines for NFCorpus
This guide walks through running BM25 with pretokenized corpus and queries and dense retrieval with BGE-Base on NFCorpus in PostgreSQL. The BM25 code follows this [script](https://github.com/castorini/quackir/blob/steven.c/psql/scripts/psql_fts_searcher.py).

## Data Prep
To fetch the data:

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip -P collections
unzip collections/nfcorpus.zip -d collections
```

To setup the data into qrels for evaluation later:

```bash
tail -n +2 collections/nfcorpus/qrels/test.tsv | sed 's/\t/\tQ0\t/' > collections/nfcorpus/qrels/test.qrels
```

To pretokenize the corpus and queries with the Lucene analyzer for consistent results:

```bash
python -m pyserini.tokenize_lucene --corpus-file collections/nfcorpus/corpus.jsonl --query-file collections/nfcorpus/queries.jsonl --output-corpus collections/nfcorpus/parsed_corpus_nfcorpus.jsonl --output-query collections/nfcorpus/parsed_queries_nfcorpus.jsonl
```

## PostgreSQL Setup
Skip this section if you already have PostgreSQL installed or know how to use it. Otherwise, create a conda environment and run the following commands to set up PostgreSQL:
```bash
conda install postgresql
initdb -D mydb
pg_ctl -D mydb -l logfile start &
createdb beir_datasets
psql
\c beir_datasets
CREATE EXTENSION vector;
\q
```

When you're done, close the database server with:
```bash
pg_ctl stop -D mydb -m smart
```

## BM25 Indexing

First, we connect to the database.
```python
import psycopg2
import json
conn = psycopg2.connect(dbname='beir_datasets', user='postgres')
cur = conn.cursor()
cur.execute("create extension vector;")
```

Then, we initialize tables for our corpus and queries.
```python
def load_jsonl_to_table(file_path, table_name):
  cur.execute(f"drop table if exists {table_name}")
  cur.execute(f"create table {table_name} (id text, contents text);")
  with open(file_path, 'r') as f:
    for line in f:
      data = json.loads(line)
      cur.execute(f"insert into {table_name} (id, contents) values (%s, %s)", (data['id'], data['contents']))
  conn.commit()
load_jsonl_to_table('collections/nfcorpus/parsed_queries_nfcorpus.jsonl', 'query')
load_jsonl_to_table('collections/nfcorpus/parsed_corpus_nfcorpus.jsonl', 'corpus')
```

Finally, we create a GIN index to speed up searching. 
```python
cur.execute('''CREATE INDEX "corpus_contents_gin" ON "corpus" USING gin(to_tsvector('simple', contents));''')
```

Now we're ready to search!

## BM25 Retrieval
Let's define a method for retrieving results for one query. We also have a helper method to clean our queries first. 
```python
import re
def _clean_query(query_string):
  cleaned = re.sub(r'[^\w\s]', ' ', query_string)
  cleaned = re.sub(r'\s+', ' ', cleaned).strip()
  return cleaned

def fts_search(query_string, top_n=1000):
  cleaned_query = _clean_query(query_string)
  ts_query = " | ".join(cleaned_query.split())
  query = f"""
  SELECT 
      id, 
      contents,
      ts_rank(to_tsvector('simple', contents), to_tsquery('simple', %s)) AS score
  FROM 
      corpus
  WHERE
      to_tsvector('simple', contents) @@ to_tsquery('simple', %s)
  ORDER BY 
      score DESC
  LIMIT %s
  """
  cur.execute(query, (ts_query, ts_query, top_n))
  return cur.fetchall()
```

We call the method on all our queries to retrieve the top 1000 results for each.
```python
from tqdm import tqdm
cur.execute("SELECT id, contents FROM query")
queries = cur.fetchall()
run_tag = "bm25_psql"

all_results = []

for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
    results = fts_search(query_string, top_n=1000)
    for rank, (doc_id, _, score) in enumerate(results, 1):
        all_results.append((query_id, doc_id, score, rank))

all_results.sort(key=lambda x: (x[0], x[3]))

with open("runs/psql_pretokenized_nfcorpus.txt", "w") as f:
    for query_id, doc_id, score, rank in all_results:
        a = f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")
```

## BGE-Base Indexing

Moving on to dense retrieval, the pipeline is very similar to what we used for BM25, except we're searching embeddings. 

We encode the corpus and queries to obtain these embeddings:

```bash
mkdir indexes/nfcorpus.bge-base-en-v1.5

python -m pyserini.encode \
  input   --corpus collections/nfcorpus/corpus.jsonl \
          --fields title text \
  output  --embeddings indexes/nfcorpus.bge-base-en-v1.5 \
  encoder --encoder BAAI/bge-base-en-v1.5 --l2-norm \
          --device cpu \
          --pooling mean \
          --fields title text \
          --batch 32

move indexes/nfcorpus.bge-base-en-v1.5/embeddings.jsonl indexes/nfcorpus.bge-base-en-v1.5/corpus_embeddings.jsonl

python -m pyserini.encode \
  input   --corpus collections/nfcorpus/queries.jsonl \
          --fields title text \
  output  --embeddings indexes/nfcorpus.bge-base-en-v1.5 \
  encoder --encoder BAAI/bge-base-en-v1.5 --l2-norm \
          --device cpu \
          --pooling mean \
          --fields title text \
          --batch 32

move indexes/nfcorpus.bge-base-en-v1.5/embeddings.jsonl indexes/nfcorpus.bge-base-en-v1.5/query_embeddings.jsonl
```

We use the ```pgvector``` extension for searching. Install it with:
```bash
conda install -c conda-forge pgvector
```

Then, like before, we connect to the database in Python. 
```python
import psycopg2
import json
conn = psycopg2.connect(dbname='beir_datasets', user='postgres')
cur = conn.cursor()
```

Then, we initialize tables for our corpus and queries.
```python
corpus_path = 'indexes/nfcorpus.bge-base-en-v1.5/corpus_embeddings.jsonl'
query_path = 'indexes/nfcorpus.bge-base-en-v1.5/query_embeddings.jsonl'

def load_jsonl_to_table(file_path, table_name, embedding_size=768):
  cur.execute(f"create table {table_name} (id text, embedding vector({embedding_size}));")
  with open(file_path, 'r') as file:
    for line in file:
      row = json.loads(line.strip())
      a = conn.execute(f"""insert into {table_name} (id, embedding) values (%s, %s)""", (row['id'], row['vector']))
  conn.commit()

embd_dim = 0
import json
with open(corpus_path, 'r') as file:
  for line in file:
    row = json.loads(line.strip())
    embd_dim = len(row['vector'])
    break

load_jsonl_to_table(corpus_path, "corpus", embd_dim)
load_jsonl_to_table(query_path, "query", embd_dim)
```

## BGE-Base Retrieval
Let's define a method for retrieving results for one query. We use ```pgvector```'s built-in distance method, specifically cosine distance, to find the closest document embeddings to our query embeddings. 
```python
def embedding_search(vector, top_n=1000):
  query = "select id, 1 - (embedding <=> %s::vector) as score from corpus order by score desc limit %s"
  cur.execute(query, (vector, top_n))
  return cur.fetchall()
```

We call the method on all our queries to retrieve the top 1000 results for each. This is almost identical to what we did for BM25, except we're passing in the query's embeddings to search for the closest document embeddings instead of searching for query contents.
```python
from tqdm import tqdm
queries = conn.execute("SELECT id, embedding FROM query").fetchall()
run_tag = "bge_psql"

all_results = []

for query_id, query_embedding in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
  results = embedding_search(query_embedding, top_n=1000)
  for rank, (doc_id, score) in enumerate(results, 1):
    all_results.append((query_id, doc_id, score, rank))

all_results.sort(key=lambda x: (x[0], x[3]))

with open("runs/psql_bge_nfcorpus.txt", "w") as f:
  for query_id, doc_id, score, rank in all_results:
    a = f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")
```

## Evaluation
To run evaluation:
```
# In project root
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/psql_pretokenized_nfcorpus.txt

python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/psql_bge_nfcorpus.txt
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.2965    |
| BGE-Base (en-v1.5)                                                                                    | 0.3808    |
> The BM25 score is lower than [Pyserini's BM25 score](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md) but reasonable considering that PostgreSQL does not implement BM25. The BGE score exactly matches [that in Pyserini](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md). 