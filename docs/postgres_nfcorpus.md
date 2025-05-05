# PostgreSQL Full Text Search Baselines for NFCorpus
This guide walks through running PostgreSQL's built-in full text search on NFCorpus with pretokenized corpus and queries. The code follows this [script](https://github.com/castorini/quackir/blob/steven.c/psql/scripts/psql_fts_searcher.py).

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
```

When you're done, close the database server with:
```bash
pg_ctl stop -D mydb -m smart
```

## Indexing

First, we connect to the database.
```python
import psycopg2
import json
conn = psycopg2.connect(dbname='beir_datasets', user='postgres')
cur = conn.cursor()
```

Then, we initialize tables for our corpus and queries.
```python
cur.execute("create table corpus (id text, contents text);")
with open('parsed_corpus_nfcorpus.jsonl', 'r') as f:
   for line in f:
     data = json.loads(line)
     cur.execute("insert into corpus (id, contents) values (%s, %s)", (data['id'], data['contents']))
conn.commit()

cur.execute("create table query (id text, contents text);")
with open('parsed_queries_nfcorpus.jsonl', 'r') as f:
   for line in f:
     data = json.loads(line)
     cur.execute("insert into query (id, contents) values (%s, %s)", (data['id'], data['contents']))
conn.commit()
```

Finally, we create a GIN index to speed up searching. 
```python
cur.execute('''CREATE INDEX "corpus_contents_gin" ON "corpus" USING gin(to_tsvector('simple', contents));''')
```

Now we're ready to search!

## Running Retrieval
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

## Evaluation
To run evaluation:
```
# In project root
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/psql_pretokenized_nfcorpus.txt
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.2965    |
> This is lower than [Pyserini's BM25 score](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md) but reasonable considering that PostgreSQL does not implement BM25.