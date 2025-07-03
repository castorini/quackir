# QuackIR: BGE-base-en-v1.5 Baselines for BEIR (v1.0.0)

This page documents BGE-base-en-v1.5 experiments for [BEIR (v1.0.0)](http://beir.ai/) using DuckDB, SQLite, and PostgreSQL.

For details walkthroughs of the implementation and setting up the RDBMS, see these [guides](../db_guides/). 

Some larger datasets are not included as the high latency makes it impractical. 

All the BEIR corpora, encoded by the BGE-base-en-v1.5 model and stored in Parquet format, are available for download:

```bash
wget https://rgw.cs.uwaterloo.ca/pyserini/data/beir-v1.0.0-bge-base-en-v1.5.parquet.tar -P collections/
tar xvf collections/beir-v1.0.0-bge-base-en-v1.5.parquet.tar -C collections/
```

The tarball is 127 GB and has MD5 checksum `5f8dce18660cc8ac0318500bea5993ac`.

## Indexing

To index all the collections:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex trec-covid webis-touche2020 quora robust04 trec-news); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Index corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index $c \
    --db-type duckdb \
    --db-path duck.db

    # Index corpus in PostgreSQL
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index $c \
    --db-type postgres 
done
```

For additional details, see explanation of [indexing](../usage-index.md).

## Retrieval

Topics and qrels are stored [here](https://github.com/castorini/anserini-tools/tree/master/topics-and-qrels), which is linked to the QuackIR repo as a submodule.

After indexing has completed, you should be able to perform retrieval as follows:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex trec-covid webis-touche2020 quora robust04 trec-news); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Retrieval with DuckDB
    python -m quackir.search \
    --topics ./tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.bge-base-en-v1.5.jsonl.gz \
    --index $c \
    --output runs/duckdb-beir-$c-dense.txt \
    --db-type duckdb \
    --db-path duck.db

    # Retrieval with PostgreSQL
    python -m quackir.search \
    --topics ./tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.bge-base-en-v1.5.jsonl.gz \
    --index $c \
    --output runs/postgres-beir-$c-dense.txt \
    --db-type postgres 
done
```

For additional details, see explanation of [search](../usage-search.md).

## Evaluation

Evaluation can be performed using Pyserini:

```bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex trec-covid webis-touche2020 quora robust04 trec-news); 
for c in "${CORPORA[@]}"
do
    echo $c

    echo "duckdb"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/duckdb-beir-$c-dense.txt

    echo "postgres"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/postgres-beir-$c-dense.txt
done
```

## Results

With the above commands, you should be able to reproduce the following results measured with nDCG@10:

| Corpus                    | DuckDB | Postgres |
| ------------------------- | ------ | -------- |
| `trec-covid`              | 0.7814 | 0.7814   |
| `bioasq`                  | \-     | \-       |
| `nfcorpus`                | 0.3735 | 0.3735   |
| `nq`                      | \-     | \-       |
| `hotpotqa`                | \-     | \-       |
| `fiqa`                    | 0.4065 | 0.4065   |
| `signal1m`                | \-     | \-       |
| `trec-news`               | 0.4425 | 0.4425   |
| `robust04`                | 0.4465 | 0.4465   |
| `arguana`                 | 0.6361 | 0.6361   |
| `webis-touche2020`        | 0.2570 | 0.2570   |
| `cqadupstack-android`     | 0.5075 | 0.5075   |
| `cqadupstack-english`     | 0.4857 | 0.4857   |
| `cqadupstack-gaming`      | 0.5965 | 0.5965   |
| `cqadupstack-gis`         | 0.4127 | 0.4127   |
| `cqadupstack-mathematica` | 0.3163 | 0.3163   |
| `cqadupstack-physics`     | 0.4722 | 0.4722   |
| `cqadupstack-programmers` | 0.4242 | 0.4242   |
| `cqadupstack-stats`       | 0.3732 | 0.3732   |
| `cqadupstack-tex`         | 0.3115 | 0.3115   |
| `cqadupstack-unix`        | 0.4219 | 0.4219   |
| `cqadupstack-webmasters`  | 0.4065 | 0.4065   |
| `cqadupstack-wordpress`   | 0.3547 | 0.3547   |
| `quora`                   | 0.8890 | 0.8890   |
| `dbpedia-entity`          | \-     | \-       |
| `scidocs`                 | 0.2170 | 0.2170   |
| `fever`                   | \-     | \-       |
| `climate-fever`           | \-     | \-       |
| `scifact`                 | 0.7408 | 0.7408   |