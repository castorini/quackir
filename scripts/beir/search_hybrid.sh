#!/bin/bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
for c in "${CORPORA[@]}"
do 
    echo $c

    # Retrieval with DuckDB
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/combined_queries/$c/queries.jsonl \
    --index $c "$c"_sparse \
    --output runs/duckdb-beir-$c-hybrid.txt \
    --pretokenized \
    --db-type duckdb \
    --db-path duck.db

    # Retrieval with PostgreSQL
    python -m quackir.search \
    --topics ./collections/beir-v1.0.0/combined_queries/$c/queries.jsonl \
    --index $c "$c"_sparse \
    --output runs/postgres-beir-$c-hybrid.txt \
    --pretokenized \
    --db-type postgres 

done

echo "done"