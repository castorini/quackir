#!/bin/bash
CORPORA=(nfcorpus scifact arguana cqadupstack-mathematica cqadupstack-webmasters cqadupstack-android scidocs cqadupstack-programmers cqadupstack-gis cqadupstack-physics cqadupstack-english cqadupstack-stats cqadupstack-gaming cqadupstack-unix cqadupstack-wordpress fiqa cqadupstack-tex);
for c in "${CORPORA[@]}"
do
    echo $c

    python scripts/combine_contents_vector.py \
    --parsed-file collections/beir-v1.0.0/corpus/$c/parsed_queries.jsonl \
    --embedding-file tools/topics-and-qrels/topics.beir-v1.0.0-$c.test.bge-base-en-v1.5.jsonl.gz \
    --output-file collections/beir-v1.0.0/combined_queries/$c/queries.jsonl

done
echo "done"