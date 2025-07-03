#!/bin/bash
# CORPORA=(trec-covid webis-touche2020 quora robust04 trec-news); 
# CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
# CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex trec-covid webis-touche2020 quora robust04 trec-news); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Index corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index "$c"_dense \
    --db-type duckdb \
    --db-path duck.db

    # Index corpus in PostgreSQL
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/bge-base-en-v1.5/$c.parquet/ \
    --index-type dense \
    --index "$c"_dense \
    --db-type postgres 
done
echo "done"