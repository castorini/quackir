#!/bin/bash
# CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
CORPORA=(trec-covid webis-touche2020 quora robust04 trec-news); 
# CORPORA=(nq signal1m dbpedia-entity hotpotqa fever climate-fever bioasq);
# CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
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

echo "done"