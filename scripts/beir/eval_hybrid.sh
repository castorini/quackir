#!/bin/bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
for c in "${CORPORA[@]}"
do 
    echo $c

    # Retrieval with DuckDB
    echo "duckdb"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/duckdb-beir-$c-hybrid.txt
    
    # Retrieval with PostgreSQL
    echo "postgres"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 tools/topics-and-qrels/qrels.beir-v1.0.0-$c.test.txt runs/postgres-beir-$c-hybrid.txt

done

echo "done"