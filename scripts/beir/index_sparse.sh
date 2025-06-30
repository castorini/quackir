#!/bin/bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Index corpus in DuckDB
    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index "$c"_sparse \
    --pretokenized \
    --db-type duckdb \
    --db-path duck.db

    python -m quackir.index \
    --input ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl \
    --index-type sparse \
    --index "$c"_sparse \
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
    --index "$c"_sparse \
    --pretokenized \
    --db-type postgres 
done
echo "done"