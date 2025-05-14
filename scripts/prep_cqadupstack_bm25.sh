#!/bin/bash

base_dir="/store/collections/beir-v1.0.0/original/cqadupstack"
for subdir in "$base_dir"/*; do
    echo $subdir

    collection_name=$(basename "$subdir")

    mkdir collections/cqadupstack-${collection_name}
    mkdir collections/cqadupstack-${collection_name}/qrels

    tail -n +2 ${subdir}/qrels/test.tsv | sed 's/\t/\tQ0\t/' > collections/cqadupstack-${collection_name}/qrels/test.qrels

    # need to push new version of tokenize script
    python ../pyserini/scripts/tokenize_lucene.py \ 
    --corpus-file ${subdir}/corpus.jsonl \
    --query-file ../anserini/tools/topics-and-qrels/topics.beir-v1.0.0-cqadupstack-${collection_name}.test.tsv.gz\
    --output-corpus collections/cqadupstack-${collection_name}/parsed_corpus_cqadupstack-${collection_name}.jsonl \
    --output-query collections/cqadupstack-${collection_name}/parsed_queries_cqadupstack-${collection_name}.jsonl
done

echo "done"