#!/bin/bash

CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Retrieval with DuckDB
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --index "$c"_sparse \
    --pretokenized \
    --output runs/duckdb-beir-$c-sparse.txt \
    --db-type duckdb \
    --db-path duck.db

    # Retrieval with SQLite
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --index "$c"_sparse \
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
    --index "$c"_sparse \
    --pretokenized \
    --output runs/postgres-beir-$c-sparse.txt \
    --db-type postgres 
done
echo "done"