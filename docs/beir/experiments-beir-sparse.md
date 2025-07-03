# QuackIR: Sparse Baselines for BEIR (v1.0.0)

This page documents sparse retrieval experiments for [BEIR (v1.0.0)](http://beir.ai/) using DuckDB, SQLite, and PostgreSQL.

DuckDB and SQLite implement BM25 while PostgreSQL uses its own implementation of full text search. 

For details walkthroughs of the implementation and setting up the RDBMS, see these [guides](../db_guides/). 

Some larger datasets are skipped for PostgreSQL as the high latency makes it impractical. 

These experiments index the corpus in a "flat" manner, by concatenating the "title" and "text" into the "contents" field.

All the original BEIR corpora are available for download:

```bash
wget https://rgw.cs.uwaterloo.ca/pyserini/data/beir-v1.0.0-corpus.tar -P collections/
tar xvf collections/beir-v1.0.0-corpus.tar -C collections/
```

The tarball is 14 GB and has MD5 checksum `faefd5281b662c72ce03d22021e4ff6b`.

## Data Prep

To munge and tokenize the corpus and queries for sparse retrieval:

```bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Tokenize and munge the corpus
    python -m quackir.analysis \
    --input ./collections/beir-v1.0.0/corpus/$c/corpus.jsonl \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl

    # Tokenize and munge the queries
    python -m quackir.analysis \
    --input ./tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.tsv.gz \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl
done
```

If you data is in an unsupported format, you can write your own script to munge it and tokenize with QuackIR during indexing.

For additional details, see explanation of [analysis](../usage-analysis.md).

## Indexing

To index all the collections:

```bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Index corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index $c \
    --pretokenized \
    --db-type duckdb \
    --db-path duck.db

    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index $c \
    --pretokenized \
    --db-type sqlite \
    --db-path sqlite.db

    # Skip certain corpora that take too long
    if [[ "$c" == "trec-covid" || "$c" == "webis-touche2020" || "$c" == "quora" || "$c" == "robust04" || "$c" == "trec-news" || "$c" == "nq" || "$c" == "signal1m" || "$c" == "dbpedia-entity" || "$c" == "hotpotqa" || "$c" == "fever" || "$c" == "climate-fever" || "$c" == "bioasq" ]]; then
      continue
    fi
    # Index corpus in PostgreSQL
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index $c \
    --pretokenized \
    --db-type postgres 
done
```

For additional details, see explanation of [indexing](../usage-index.md).

## Retrieval

Topics and qrels are stored [here](https://github.com/castorini/anserini-tools/tree/master/topics-and-qrels), which is linked to the QuackIR repo as a submodule.

After indexing has completed, you should be able to perform retrieval as follows:

```bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Retrieval with DuckDB
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --index $c \
    --pretokenized \
    --output runs/duckdb-beir-$c-sparse.txt \
    --db-type duckdb \
    --db-path duck.db

    # Retrieval with SQLite
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --index $c \
    --pretokenized \
    --output runs/sqlite-beir-$c-sparse.txt \
    --db-type sqlite \
    --db-path sqlite.db

    # Skip certain corpora that take too long
    if [[ "$c" == "trec-covid" || "$c" == "webis-touche2020" || "$c" == "quora" || "$c" == "robust04" || "$c" == "trec-news" || "$c" == "nq" || "$c" == "signal1m" || "$c" == "dbpedia-entity" || "$c" == "hotpotqa" || "$c" == "fever" || "$c" == "climate-fever" || "$c" == "bioasq" ]]; then
      continue
    fi
    # Retrieval with PostgreSQL
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --index $c \
    --pretokenized \
    --output runs/postgres-beir-$c-sparse.txt \
    --db-type postgres 
done
```

For additional details, see explanation of [search](../usage-search.md).

## Evaluation

Evaluation can be performed using Pyserini:

```bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    echo "duckdb"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/duckdb-beir-$c-sparse.txt

    echo "sqlite"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ./tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/sqlite-beir-$c-sparse.txt

    # Skip certain corpora that take too long
    if [[ "$c" == "trec-covid" || "$c" == "webis-touche2020" || "$c" == "quora" || "$c" == "robust04" || "$c" == "trec-news" || "$c" == "nq" || "$c" == "signal1m" || "$c" == "dbpedia-entity" || "$c" == "hotpotqa" || "$c" == "fever" || "$c" == "climate-fever" || "$c" == "bioasq" ]]; then
      continue
    fi
    echo "postgres"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/postgres-beir-$c-sparse.txt
done
```

## Results

With the above commands, you should be able to reproduce the following results measured with nDCG@10:

| Corpus                      | DuckDB | SQLite | PostgreSQL |
| --------------------------- | ------ | ------ | ---------- |
| \`trec-covid\`              | 0.5947 | 0.6011 | \-         |
| \`bioasq\`                  | 0.5210 | 0.5130 | \-         |
| \`nfcorpus\`                | 0.3206 | 0.3223 | 0.2965     |
| \`nq\`                      | 0.3050 | 0.2921 | \-         |
| \`hotpotqa\`                | 0.6357 | 0.5933 | \-         |
| \`fiqa\`                    | 0.2378 | 0.2518 | 0.0918     |
| \`signal1m\`                | 0.3396 | 0.3308 | \-         |
| \`trec-news\`               | 0.3849 | 0.4031 | \-         |
| \`robust04\`                | 0.4081 | 0.4243 | \-         |
| \`arguana\`                 | 0.3179 | 0.4806 | 0.0690     |
| \`webis-touche2020\`        | 0.4352 | 0.3471 | \-         |
| \`cqadupstack-android\`     | 0.3812 | 0.3942 | 0.2607     |
| \`cqadupstack-english\`     | 0.3441 | 0.3672 | 0.2252     |
| \`cqadupstack-gaming\`      | 0.4827 | 0.4876 | 0.3436     |
| \`cqadupstack-gis\`         | 0.2893 | 0.3002 | 0.1864     |
| \`cqadupstack-mathematica\` | 0.2036 | 0.2185 | 0.1215     |
| \`cqadupstack-physics\`     | 0.3213 | 0.3474 | 0.2053     |
| \`cqadupstack-programmers\` | 0.2803 | 0.2965 | 0.1866     |
| \`cqadupstack-stats\`       | 0.2728 | 0.2838 | 0.1828     |
| \`cqadupstack-tex\`         | 0.2256 | 0.2419 | 0.1303     |
| \`cqadupstack-unix\`        | 0.2779 | 0.2869 | 0.1678     |
| \`cqadupstack-webmasters\`  | 0.3070 | 0.3078 | 0.2319     |
| \`cqadupstack-wordpress\`   | 0.2485 | 0.2579 | 0.1280     |
| \`quora\`                   | 0.7893 | 0.8063 | \-         |
| \`dbpedia-entity\`          | 0.3177 | 0.3191 | \-         |
| \`scidocs\`                 | 0.1502 | 0.1542 | 0.0907     |
| \`fever\`                   | 0.6475 | 0.5590 | \-         |
| \`climate-fever\`           | 0.1486 | 0.1335 | \-         |
| \`scifact\`                 | 0.6795 | 0.6862 | 0.5692     |