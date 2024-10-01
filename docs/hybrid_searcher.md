# Hybrid Searcher Baseline for NFCorpus
This guide generally mirrors the guide [here](https://motherduck.com/blog/search-using-duckdb-part-3/), except we do it on the NFCorpus dataset and use BGE-base as the dense encoder.

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

## Indexing
The indexing for BM25 is done in-memory when running retrieval. 

For BGE-base, we run:
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

## Running Retrieval
The hybrid search is abstracted in a `HybridSearcher` class in `scripts/hybrid_searcher.py`. Currently, the same file also contains a main class and acts as a script (could refactor this later), which runs:
- BM25 with `k=1.2`, `b=0.75`
- BGE-base
- Reciprocal Ranked Fusion with `k=60`
- Convex Combination with `alpha=0.8` (weights dense score 0.8, BM25 score 0.2)

Simply run the command:
```bash
cd scripts
python hybrid_searcher.py
```

>Note: When running BM25, DuckDB assigns `NULL` to all documents that have no term intersection with the query. For these documents, we have dictated that they have a BM25 score of 0.

## Evaluation
To run evaluation:
```
# In project root
python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels   runs/nfcorpus/run.bm25.txt 

python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels   runs/nfcorpus/run.bge-base-en-v1.5.txt

python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels   runs/nfcorpus/run.rrf-60.txt

python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels   runs/nfcorpus/run.cc-0.8.txt 
```
which should yield:

| **Retrieval Method**                                                                                                  | **nDCG@10**  |
|:-------------------------------------------------------------------------------------------------------------|-----------|
| BM25                                                                                    | 0.3178    |
| BGE-Base (en-v1.5)                                                                                                   | 0.3808  |
| RRF (k=60)                                                                                     | 0.3594    |
| CC (alpha=0.8)                                                                                                   | 0.3587  |
> Note: The BGE-Base value matches [that in Pyserini](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md). However, the BM25 value is slightly below [that in Pyserini](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md). This could be due to how DuckDB uses different stemmers, stopwords, pre-processing (removal of accents, punctuations, numbers) than Pyserini/Anserini. For more information, see [this blog post by DuckDB](https://duckdb.org/2021/01/25/full-text-search.html).