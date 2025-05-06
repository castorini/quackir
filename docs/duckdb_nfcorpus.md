# DuckDB Baselines for NFCorpus
This guide walks through running BM25 and dense retrieval with BGE-Base on NFCorpus in DuckDB with pretokenized corpus and queries. The BM25 code follows this [script](https://github.com/castorini/quackir/blob/vivek.a/duckDB-experimentation/scripts/bm25_benchmarking.py) and the BGE-base code follows this [script](https://github.com/castorini/quackir/blob/main/scripts/hybrid_searcher.py).

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

For BM25, we pretokenize the corpus and queries with the Lucene analyzer for consistent results:

```bash
python ../pyserini/scripts/tokenize_lucene --corpus-file collections/nfcorpus/corpus.jsonl --query-file collections/nfcorpus/queries.jsonl --output-corpus collections/nfcorpus/parsed_corpus_nfcorpus.jsonl --output-query collections/nfcorpus/parsed_queries_nfcorpus.jsonl
```

## BM25 Indexing
Let's start with BM25. 
First, we connect to a DB file.
```python
import duckdb
conn = duckdb.connect("duck.db")
```

Then, we initialize tables for our corpus and queries. DuckDB provides convenient methods for reading from certain file types.
```python
conn.execute(f"""CREATE TABLE corpus AS SELECT id AS id, contents AS contents FROM read_json('collections/nfcorpus/parsed_corpus_nfcorpus.jsonl', format = 'newline_delimited');""")
conn.execute(f"""CREATE TABLE query AS SELECT id AS id, contents AS contents FROM read_json('collections/nfcorpus/parsed_queries_nfcorpus.jsonl', format = 'newline_delimited');""")
```

Finally, we create the BM25 index. Note we turn off all the built-in tokenization options since we're using pretokenized data to avoid performance differences due to tokenization. 
```python
conn.execute("PRAGMA create_fts_index(corpus, id, contents, stemmer = 'none', stopwords = 'none', ignore = 'a^', strip_accents = 0, lower = 0)")
```

Now we're ready to search!

## BM25 Retrieval
Let's define a method for retrieving results for one query. Note the use of ```COALESCE``` to turn null values into zeroes as DuckDB sets the score to ```NULL``` when no terms from the query appear in the document. 
```python
def fts_search(query_string, top_n=5):
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
    return conn.execute(query, [query_string, top_n]).fetchall()
```

We call the method on all our queries to retrieve the top 1000 results for each.
```python
from tqdm import tqdm
queries = conn.execute("SELECT id, contents FROM query").fetchall()
run_tag = "bm25_duckdb"

all_results = []

for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
    results = fts_search(query_string, top_n=1000)
    for rank, (doc_id, _, score) in enumerate(results, 1):
        all_results.append((query_id, doc_id, score, rank))

all_results.sort(key=lambda x: (x[0], x[3])) # sort by queryid, then rank

with open("runs/duckdb_pretokenized_nfcorpus.txt", "w") as f:
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

Then, we can connect to the database in Python. 
```python
import duckdb
conn = duckdb.connect(":memory:")
```

Now we initialize and load tables for our corpus and queries. We use DuckDB's float array to hold our embeddings.
```python
corpus_path = 'indexes/nfcorpus.bge-base-en-v1.5/corpus_embeddings.jsonl'
query_path = 'indexes/nfcorpus.bge-base-en-v1.5/query_embeddings.jsonl'

embd_dim = 0
import json
with open(corpus_path, 'r') as file:
    for line in file:
        row = json.loads(line.strip())
        embd_dim = len(row['vector'])
        break

conn.execute(f"""create table corpus (id varchar primary key, contents varchar, embedding float[{embd_dim}])""")
conn.execute(f"""create table query (id varchar primary key, contents varchar, embedding float[{embd_dim}])""")

def load_jsonl_to_table(file_path, table_name):
    with open(file_path, 'r') as file:
        for line in file:
            row = json.loads(line.strip())
            a = conn.execute(f"""insert into {table_name} (id, contents, embedding) values (?, ?, ?)""", (row['id'], row['contents'], row['vector']))

load_jsonl_to_table(corpus_file, "corpus")
load_jsonl_to_table(query_file, "query")
```

## BGE-Base Retrieval
Let's define a method for retrieving results for one query. We use DuckDB's ```array_cosine_similarity``` method to find the closest document embeddings to our query embeddings. 
```python
def embedding_search(query_id, top_n=5):
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
    return conn.execute(query, [query_id, top_n]).fetchall()
```

We call the method on all our queries to retrieve the top 1000 results for each. This is almost identical to what we did for BM25, except we're passing in the query ID to the search method to get the appropriate embeddings instead of searching the query contents directly.
```python
from tqdm import tqdm
queries = conn.execute("SELECT id, contents FROM query").fetchall()
run_tag = "bge_duckdb"

all_results = []

for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
    results = embedding_search(query_id, top_n=1000)
    for rank, (doc_id, _, score) in enumerate(results, 1):
        all_results.append((query_id, doc_id, score, rank))

all_results.sort(key=lambda x: (x[0], x[3])) # sort by queryid, then rank

with open("runs/duckdb_bge_nfcorpus.txt", "w") as f:
    for query_id, doc_id, score, rank in all_results:
        a = f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")
```

## Evaluation
To run evaluation:
```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/duckdb_pretokenized_nfcorpus.txt

python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/duckdb_bge_nfcorpus.txt
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.3218    |
| BGE-Base (en-v1.5)                                                                                    | 0.3808    |
> The [BM25](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md) and [BGE](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md) scores both exactly matches that in Pyserini.