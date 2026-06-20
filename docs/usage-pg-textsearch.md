# Using Timescale pg_textsearch (BM25) with QuackIR

This guide shows how to run BM25 keyword search in PostgreSQL/Timescale using the `pg_textsearch` extension, integrated into QuackIR’s Postgres back end.

Note: This guide assumes you have already preprocessed NFCorpus as in the experiments guide (both sparse text and dense embeddings).
See [experiments-nfcorpus.md](./experiments-nfcorpus.md). If not, please follow that first.

## Prerequisites

1. Create a Tiger Data account: https://console.cloud.timescale.com/signup
2. Find your service connection details: https://docs.tigerdata.com/integrations/latest/find-connection-details/
3. At the root of this repo, create a `.env` file with your Timescale DSN (service URL):

```
TIMESCALE_SERVICE_URL=postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DBNAME>?sslmode=require
```

## Indexing with BM25

- `use_pg_textsearch=True` switches QuackIR to use Timescale’s `pg_textsearch` BM25 index instead of Postgres GIN/tsvector.
- You can tune BM25 via `k1` (term frequency saturation) and `b` (length normalization).

```python
from quackir.index import PostgresIndexer
from quackir import IndexType

table_name = "corpus"
index_type = IndexType.SPARSE
corpus_file = "collections/nfcorpus/quackir_corpus.jsonl"

indexer = PostgresIndexer(use_pg_textsearch=True)
indexer.init_table(table_name, index_type)
indexer.load_table(table_name, corpus_file, pretokenized=True)
indexer.fts_index(table_name, k1=0.9, b=0.4)

indexer.close()
```

## Searching with BM25

- When `use_pg_textsearch=True`, QuackIR runs `pg_textsearch` BM25 queries via the distance operator `<@>` and `to_bm25query(...)`.

```python
from quackir.search import PostgresSearcher
from quackir import SearchType
import csv
import pathlib

table_name = "corpus"
top_k = 1000

searcher = PostgresSearcher(use_pg_textsearch=True)

with pathlib.Path("runs/run.quackir.postgres.sparse.pg_textsearch.nfcorpus.txt").open(
    "w"
) as out:
    with open("collections/nfcorpus/queries.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        for qid, qtext in r:
            hits = searcher.search(
                SearchType.SPARSE,
                query_string=qtext,
                table_names=[table_name],
                top_n=top_k,
            )
            buf = "".join(
                f"{qid} Q0 {docid} {rank} {-score:.6f} QuackIR\n"
                for rank, (docid, score) in enumerate(hits, 1)
            )
            out.write(buf)

searcher.close()
```

### Why negate the score?
`pg_textsearch` returns BM25 scores as negative values so that lower (more negative) is better.
This choice aligns naturally with Postgres’s default ascending order for distance/score operations.
If your downstream code or evaluation expects larger scores to indicate better matches, simply flip the sign when writing results as shown above.

## Evaluating results

After the run finishes, we can evaluate the results using `trec_eval`:

```bash
python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
  runs/run.quackir.postgres.sparse.pg_textsearch.nfcorpus.txt
```

which should yield:

```
ndcg_cut_10             all     0.3084
```

## Optional: Hybrid search (BM25 + embeddings)

The following script demonstrates hybrid search that fuses BM25 (`pg_textsearch`) with dense vector similarity using RRF.

```python
from quackir.index import PostgresIndexer
from quackir.search import PostgresSearcher
from quackir import IndexType, SearchType
import pathlib
import json

"""
Index
"""

sparse_table_name = "corpus"
dense_table_name = "corpus_dense"
corpus_file = "collections/nfcorpus/quackir_corpus.jsonl"
embedding_file = "indexes/nfcorpus.bge-base-en-v1.5/embeddings.parquet"
embedding_dim = 768

indexer = PostgresIndexer(use_pg_textsearch=True)

indexer.init_table(sparse_table_name, IndexType.SPARSE)
indexer.load_table(sparse_table_name, corpus_file, IndexType.SPARSE, pretokenized=True)
indexer.init_table(dense_table_name, IndexType.DENSE, embedding_dim=embedding_dim)
indexer.load_table(dense_table_name, embedding_file, IndexType.DENSE)

indexer.fts_index(sparse_table_name, text_config="english")
indexer.embedding_index(
    dense_table_name, using="hnsw", opclass="vector_cosine_ops", column="embedding"
)

indexer.close()

"""
Search
"""

searcher = PostgresSearcher(use_pg_textsearch=True)

top_k = 1000
output_file = pathlib.Path(
    "runs/run.quackir.postgres.hybrid.pg_textsearch.nfcorpus.txt"
)
query_embedding_file = pathlib.Path(
    "collections/nfcorpus/queries.bge-base-en-v1.5/embeddings.jsonl"
)

with output_file.open("w") as out:
    with query_embedding_file.open() as f:
        for line in f:
            query = json.loads(line)
            qid = query["id"]
            qtext = query["contents"]
            qvector = query["vector"]
            hits = searcher.search(
                SearchType.HYBRID,
                query_string=qtext,
                query_embedding=qvector,
                table_names=[sparse_table_name, dense_table_name],
                top_n=top_k,
                tokenize_query=False,
            )
            buf = "".join(
                f"{qid} Q0 {docid} {rank} {score:.6f} QuackIR\n"
                for rank, (docid, score) in enumerate(hits, 1)
            )
            out.write(buf)

searcher.close()
```

Then, evaluate the hybrid results:

```bash
python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
  runs/run.quackir.postgres.hybrid.pg_textsearch.nfcorpus.txt
```

which should yield:

```
ndcg_cut_10             all     0.3718
```

## Comparison

We can perform a sparse search using the default Postgres GIN/tsvector indexing by omitting the `use_pg_textsearch=True` flag.
You can do this on your own and evaluate the results similarly.
We then compare all the results as below:

| Method | System | Score (NDCG@10) |
| :--- | :--- | :--- |
| Sparse (GIN/tsvector) | Postgres | 0.1989 |
| Sparse (BM25) | pg_textsearch (extension) | 0.3084 |
| Hybrid (BM25 + Dense) | pg_textsearch (extension) | 0.3718 |

