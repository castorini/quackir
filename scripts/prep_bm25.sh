#!/bin/bash

CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    mkdir collections/${c}
    mkdir collections/${c}/qrels

    tail -n +2 /store/collections/beir-v1.0.0/original/${c}/qrels/test.tsv | sed 's/\t/\tQ0\t/' > collections/${c}/qrels/test.qrels

    # need to push new version of tokenize script
    python ../pyserini/scripts/tokenize_lucene.py \
    --corpus-file /store/collections/beir-v1.0.0/original/${c}/corpus.jsonl \
    --query-file ../anserini/tools/topics-and-qrels/topics.beir-v1.0.0-${c}.test.tsv.gz\
    --output-corpus collections/${c}/parsed_corpus_${c}.jsonl \
    --output-query collections/${c}/parsed_queries_${c}.jsonl
done

echo "done"