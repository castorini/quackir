#!/bin/bash

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