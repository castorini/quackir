# DuckDB BM25 Baselines for NFCorpus
This guide walks through running BM25 on NFCorpus in DuckDB with pretokenized corpus and queries. The code follows this [script](https://github.com/castorini/quackir/blob/vivek.a/duckDB-experimentation/scripts/bm25_benchmarking.py).

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

First, we connect to a DB file instead of a memory connection for speed.
```python
import duckdb
conn = duckdb.connect("duck.db")
```

Then, we initialize tables for our corpus and queries. DuckDB provides convenient methods for reading from certain file types.
```python
conn.execute(f"""CREATE TABLE corpus AS SELECT id AS id, contents AS contents FROM read_json('collections/nfcorpus/parsed_corpus.jsonl', format = 'newline_delimited');""")
conn.execute(f"""CREATE TABLE query AS SELECT id AS id, contents AS contents FROM read_json('collections/nfcorpus/parsed_queries.jsonl', format = 'newline_delimited');""")
```

Finally, we create the BM25 index. Note we turn off all the built-in tokenization options since we're using pretokenized data to avoid performance differences due to tokenization. 
```python
conn.execute("PRAGMA create_fts_index(corpus, id, contents, stemmer = 'none', stopwords = 'none', ignore = 'a^', strip_accents = 0, lower = 0)")
```

Now we're ready to search!

## Running Retrieval
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

## Evaluation
To run evaluation:
```
# In project root
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels runs/duckdb_pretokenized_nfcorpus.txt
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.3218    |
> This exactly matches [that in Pyserini](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md).