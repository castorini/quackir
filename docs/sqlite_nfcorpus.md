# SQLite BM25 Baselines for NFCorpus
This guide walks through running BM25 on NFCorpus in SQLite with pretokenized corpus and queries. The code follows this [script](https://github.com/castorini/quackir/blob/vivek.a/duckDB-experimentation/scripts/bm25_benchmarking.py).

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

## Indexing

First, we connect to a DB file.
```python
import sqlite3
conn = sqlite3.connect("sqlite.db")
```

Then, we initialize tables for our corpus and queries.
```python
import json
def load_jsonl_to_table(file_path, table_name):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            contents TEXT
        )
    """)
    with open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            row_data = json.loads(line.strip())
            conn.execute(
                f"""
                INSERT INTO {table_name} (id, contents) VALUES (?, ?)
            """,
                (row_data["id"], row_data["contents"]),
            )
load_jsonl_to_table('collections/nfcorpus/parsed_corpus_nfcorpus.jsonl', 'corpus')
load_jsonl_to_table('collections/nfcorpus/parsed_queries_nfcorpus.jsonl', 'query')
```

Finally, we create the BM25 index. Note SQLite does not offer turning off tokenization, so we use the porter tokenizer with the hope that it doesn't do much since Lucene has already gone through the data with it, since we're using pretokenized data to avoid performance differences due to tokenization. Ideally, we should write our own tokenizer to make sure it actually doesn't do anything other than split the text by spaces. 
```python
conn.execute("""
    CREATE VIRTUAL TABLE fts_corpus USING fts5(
        id, contents,
        content='corpus', content_rowid='rowid', tokenize = 'porter' 
    )
""")

conn.execute(
    "INSERT INTO fts_corpus (id, contents) SELECT id, contents FROM corpus;"
)
conn.commit()
```

Now we're ready to search!

## Running Retrieval
Let's define a method for retrieving results for one query. Note we must clean our queries first. 
```python
def fts_search(query_string, top_n=5):
    query = """
    SELECT id, contents, bm25(fts_corpus)*-1 AS score
    FROM fts_corpus
    WHERE fts_corpus MATCH '{}'
    ORDER BY score DESC
    LIMIT {}
    """
    query_string = query_string.replace("'", "''").replace('"', '""')
    terms = query_string.split()
    escaped_terms = terms
    # make each term a string so that any special chars are escaped
    escaped_terms = [f'"{term}"' for term in escaped_terms]
    # allow matching of any of the terms, using + or AND will turn it into boolean AND retrieval
    res = " OR ".join(escaped_terms)
    query = query.format(res, top_n)
    return conn.execute(query).fetchall()
```

We call the method on all our queries to retrieve the top 1000 results for each.
```python
from tqdm import tqdm
queries = conn.execute("SELECT id, contents FROM query").fetchall()
run_tag = "bm25_sqlite"

all_results = []

for query_id, query_string in tqdm(queries, desc=f"Processing {run_tag}", unit="query"):
    results = fts_search(query_string, top_n=1000)
    for rank, (doc_id, _, score) in enumerate(results, 1):
        all_results.append((query_id, doc_id, score, rank))

all_results.sort(key=lambda x: (x[0], x[3])) # sort by queryid, then rank

with open("runs/sqlite_pretokenized_nfcorpus.txt", "w") as f:
    for query_id, doc_id, score, rank in all_results:
        a = f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_tag}\n")
```

## Evaluation
To run evaluation:
```
# In project root
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/sqlite_pretokenized_nfcorpus.txt
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.3218    |
> This is very close to [that in Pyserini](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md).