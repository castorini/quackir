#!/bin/bash
# CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
CORPORA=(trec-covid bioasq nfcorpus nq hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress quora dbpedia-entity scidocs fever climate-fever scifact); 
for c in "${CORPORA[@]}"
do
    echo $c

    # Tokenize and munge the corpus
    python -m quackir.analysis \
    --input ./collections/beir-v1.0.0/corpus/$c/corpus.jsonl.gz \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_corpus.jsonl

    # Tokenize and munge the queries
    python -m quackir.analysis \
    --input ./tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.tsv.gz \
    --output ./collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl
done
echo "done"