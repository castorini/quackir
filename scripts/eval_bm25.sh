#!/bin/bash
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    echo "duckdb"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/${c}/qrels/test.qrels runs/duckdb_bm25_${c}.txt
    
    echo "sqlite"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/${c}/qrels/test.qrels runs/sqlite_bm25_${c}.txt
    
    echo "postgres"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 collections/${c}/qrels/test.qrels runs/postgres_fts_${c}.txt
done

echo "done"