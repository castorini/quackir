#!/bin/bash

# Base directory for collections
base_dir="/store/collections/beir-v1.0.0/original"
qrels_dir="/store/scratch/valamuri/anserini/tools/topics-and-qrels/"
eval_dir="evals"

db_type=${1:-"sqlite"}
echo "Running script for $db_type"
# Create evals directory if it doesn't exist
mkdir -p "$eval_dir"

# Iterate over subdirectories in the base directory
for subdir in "$base_dir"/*; do
  # Skip if it's not a directory or if the name ends with .zip
  if [[ ! -d "$subdir" || "$subdir" == *.zip ]]; then
    continue
  fi

  # Get the subdirectory name
  collection_name=$(basename "$subdir")
  echo "Evaluating $collection_name"
  # Special case for cqadupstack
  if [[ "$collection_name" == "cqadupstack" ]]; then
    for subsubdir in "$subdir"/*; do
      if [[ -d "$subsubdir" ]]; then
        subcollection_name=$(basename "$subsubdir")
        run_file="runs/${db_type}_bm25_beir_cqadupstack_${subcollection_name}.txt"
        eval_file="$eval_dir/${db_type}_bm25_beir_cqadupstack_${subcollection_name}.txt"

        # Check if the runs file exists
        if [[ ! -f "$run_file" ]]; then
          echo "Run file does not exist for cqadupstack/$subcollection_name. Skipping..."
          continue
        fi

        qrels_file="$qrels_dir/qrels.beir-v1.0.0-$collection_name-$subcollection_name.test.txt"
        python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$qrels_file" "$run_file" >"$eval_file"
      fi
    done
  else
    # Handle other collections
    run_file="runs/${db_type}_bm25_beir_${collection_name}.txt"
    eval_file="$eval_dir/${db_type}_bm25_beir_${collection_name}.txt"

    # Check if the runs file exists
    if [[ ! -f "$run_file" ]]; then
      echo "Run file does not exist for $collection_name. Skipping..."
      continue
    fi

    qrels_file="$qrels_dir/qrels.beir-v1.0.0-$collection_name.test.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$qrels_file" "$run_file" >"$eval_file"
  fi
done

echo "All collections processed."
