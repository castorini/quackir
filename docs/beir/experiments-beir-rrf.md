# QuackIR: RRF Baselines for BEIR (v1.0.0)

This page documents RRF experiments for [BEIR (v1.0.0)](http://beir.ai/) using DuckDB and PostgreSQL.

For details walkthroughs of the implementation and setting up the RDBMS, see these [guides](../db_guides/). 

Some larger datasets are not included as the high latency makes it impractical. 

These experiments performs RRF of sparse and dense retrieval results. 
The sparse corpus was indexed in a "flat" manner, by concatenating the "title" and "text" into the "contents" field.
The dense retrieval uses corpus and queries encoded by the BGE-base-en-v1.5 model.

All the BEIR corpora are available for download:

```bash
wget https://rgw.cs.uwaterloo.ca/pyserini/data/beir-v1.0.0-corpus.tar -P collections/
tar xvf collections/beir-v1.0.0-corpus.tar -C collections/
```

The tarball is 14 GB and has MD5 checksum `faefd5281b662c72ce03d22021e4ff6b`.

All the BEIR corpora, encoded by the BGE-base-en-v1.5 model and stored in Parquet format, are available for download:

```bash
wget https://rgw.cs.uwaterloo.ca/pyserini/data/beir-v1.0.0-bge-base-en-v1.5.parquet.tar -P collections/
tar xvf collections/beir-v1.0.0-bge-base-en-v1.5.parquet.tar -C collections/
```

The tarball is 127 GB and has MD5 checksum `5f8dce18660cc8ac0318500bea5993ac`.

## Data Prep

To munge and tokenize the corpus and queries:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Tokenize and munge the corpus
    python -m quackir.analysis \
    --input ./collections/beir-v1.0.0/corpus/$c/ \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl

    # Tokenize and munge the queries
    python -m quackir.analysis \
    --input ./tools/topics-and-qrels/topics.beir-v1.0.0-nfcorpus.test.tsv.gz \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl

    # Combine parsed queries and query embeddings into one file
    python scripts/combine_contents_vector.py \
    --parsed-file collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --embedding-file tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.bge-base-en-v1.5.jsonl.gz \
    --output-file collections/beir-v1.0.0/combined_queries/$c/queries.jsonl
done
```

If you data is in an unsupported format, you can write your own script to munge it and tokenize with QuackIR during indexing.

For additional details, see explanation of [analysis](../usage-analysis.md).

## Indexing

To index all the collections:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Index sparse corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index "$c"_sparse \
    --pretokenized \
    --db-type duckdb \
    --db-path duck.db

    # Index dense corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index "$c"_dense \
    --db-type duckdb \
    --db-path duck.db

    # Index sparse corpus in PostgreSQL
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index "$c"_sparse \
    --pretokenized \
    --db-type postgres 

    # Index dense corpus in PostgreSQL
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index "$c"_dense \
    --db-type postgres 
done
```

For additional details, see explanation of [indexing](../usage-index.md).

## Retrieval

Topics and qrels are stored [here](https://github.com/castorini/anserini-tools/tree/master/topics-and-qrels), which is linked to the QuackIR repo as a submodule.

After indexing has completed, you should be able to perform retrieval as follows:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Retrieval with DuckDB
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/combined_queries/$c/queries.jsonl \
    --index "$c"_sparse "$c"_dense \
    --pretokenized \
    --output runs/duckdb-beir-$c-hybrid.txt \
    --db-type duckdb \
    --db-path duck.db

    # Retrieval with PostgreSQL
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/combined_queries/$c/queries.jsonl \
    --index "$c"_sparse "$c"_dense \
    --pretokenized \
    --output runs/postgres-beir-$c-hybrid.txt \
    --db-type postgres 
done
```

For additional details, see explanation of [search](../usage-search.md).

## Evaluation

Evaluation can be performed using Pyserini:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex); 
for c in "${CORPORA[@]}"
do
    echo $c

    echo "duckdb"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/duckdb-beir-$c-hybrid.txt

    echo "postgres"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/postgres-beir-$c-hybrid.txt
done
```

## Results

With the above commands, you should be able to reproduce the following results measured with nDCG@10:

| Corpus                    | DuckDB | PostgreSQL |
| ------------------------- | ------ | ---------- |
| `trec-covid`              | \-     | \-         |
| `bioasq`                  | \-     | \-         |
| `nfcorpus`                | 0.3621 | 0.3626     |
| `nq`                      | \-     | \-         |
| `hotpotqa`                | \-     | \-         |
| `fiqa`                    | 0.3683 | 0.2881     |
| `signal1m`                | \-     | \-         |
| `trec-news`               | \-     | \-         |
| `robust04`                | \-     | \-         |
| `arguana`                 | 0.5063 | 0.3449     |
| `webis-touche2020`        | \-     | \-         |
| `cqadupstack-android`     | 0.4653 | 0.4117     |
| `cqadupstack-english`     | 0.4436 | 0.3913     |
| `cqadupstack-gaming`      | 0.5628 | 0.5022     |
| `cqadupstack-gis`         | 0.3683 | 0.3290     |
| `cqadupstack-mathematica` | 0.2744 | 0.2325     |
| `cqadupstack-physics`     | 0.4138 | 0.3593     |
| `cqadupstack-programmers` | 0.3732 | 0.3293     |
| `cqadupstack-stats`       | 0.3400 | 0.3130     |
| `cqadupstack-tex`         | 0.2930 | 0.2581     |
| `cqadupstack-unix`        | 0.3620 | 0.3302     |
| `cqadupstack-webmasters`  | 0.3723 | 0.3391     |
| `cqadupstack-wordpress`   | 0.3362 | 0.2805     |
| `quora`                   | \-     | \-         |
| `dbpedia-entity`          | \-     | \-         |
| `scidocs`                 | 0.1943 | 0.1750     |
| `fever`                   | \-     | \-         |
| `climate-fever`           | \-     | \-         |
| `scifact`                 | 0.7440 | 0.6800     |