#!/bin/bash

./scripts/beir/tokenize.sh > logs/tokenize.txt
echo "tokenized"

./scripts/beir/index_sparse.sh > logs/index_sparse.txt
echo "sparse indexed"

./scripts/beir/index_bge.sh > logs/index_bge.txt
echo "bge indexed"

./scripts/beir/combine.sh > logs/combine.txt
echo "hybrid combined"

./scripts/beir/search_sparse.sh > logs/search_sparse.txt
echo "sparse retrieved"

./scripts/beir/search_bge.sh > logs/search_bge.txt
echo "bge retrieved"

./scripts/beir/search_hybrid.sh > logs/search_hybrid.txt
echo "hybrid retrieved"

./scripts/beir/eval_sparse.sh > logs/eval_sparse.txt
echo "sparse evaluated"

./scripts/beir/eval_bge.sh > logs/eval_bge.txt
echo "bge evaluated"

./scripts/beir/eval_hybrid.sh > logs/eval_hybrid.txt
echo "hybrid evaluated"

echo "done"