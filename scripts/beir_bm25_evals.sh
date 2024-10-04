#!/bin/bash

# Base directory for collections
base_dir="/store/collections/beir-v1.0.0/original"
output_dir="collections"
eval_dir="evals"

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

  # Special case for cqadupstack
  if [[ "$collection_name" == "cqadupstack" ]]; then
    for subsubdir in "$subdir"/*; do
      if [[ -d "$subsubdir" ]]; then
        subcollection_name=$(basename "$subsubdir")
        run_file="runs/bm25_beir_cqadupstack_${subcollection_name}.txt"
        eval_file="$eval_dir/bm25_beir_cqadupstack_${subcollection_name}.txt"
        
        # Check if the runs file exists
        if [[ ! -f "$run_file" ]]; then
          echo "Run file does not exist for cqadupstack/$subcollection_name. Skipping..."
          continue
        fi
        
        # Create qrels directory
        mkdir -p "$output_dir/cqadupstack/$subcollection_name/qrels"
        
        # Process qrels file and run evaluation
        tail -n +2 "$subsubdir/qrels/test.tsv" | sed 's/\t/\tQ0\t/' > "$output_dir/cqadupstack/$subcollection_name/qrels/test.qrels"
        python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$output_dir/cqadupstack/$subcollection_name/qrels/test.qrels" "$run_file" > "$eval_file"
      fi
    done
  else
    # Handle other collections
    run_file="runs/bm25_beir_${collection_name}.txt"
    eval_file="$eval_dir/bm25_beir_${collection_name}.txt"

    # Check if the runs file exists
    if [[ ! -f "$run_file" ]]; then
      echo "Run file does not exist for $collection_name. Skipping..."
      continue
    fi
    
    # Create qrels directory
    mkdir -p "$output_dir/$collection_name/qrels"
    
    # Process qrels file and run evaluation
    tail -n +2 "$subdir/qrels/test.tsv" | sed 's/\t/\tQ0\t/' > "$output_dir/$collection_name/qrels/test.qrels"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$output_dir/$collection_name/qrels/test.qrels" "$run_file" > "$eval_file"
  fi
done

echo "All collections processed."

