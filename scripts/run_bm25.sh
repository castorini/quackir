#!/bin/bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    echo "duckdb"
    python src/search_dbs.py \
    --corpus-file collections/${c}/parsed_corpus_${c}.jsonl \
    --query-file collections/${c}/parsed_queries_${c}.jsonl \
    --output-file runs/duckdb_bm25_${c}.txt \
    --db duckdb \
    --method fts

    echo "sqlite"
    python src/search_dbs.py \
    --corpus-file collections/${c}/parsed_corpus_${c}.jsonl \
    --query-file collections/${c}/parsed_queries_${c}.jsonl \
    --output-file runs/sqlite_bm25_${c}.txt \
    --db sqlite \
    --method fts

    echo "postgres"
    python src/search_dbs.py \
    --corpus-file collections/${c}/parsed_corpus_${c}.jsonl \
    --query-file collections/${c}/parsed_queries_${c}.jsonl \
    --output-file runs/postgres_fts_${c}.txt \
    --db postgres \
    --method fts
done

echo "done"