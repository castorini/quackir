#!/bin/bash

# Base directory
base_dir="/store/collections/beir-v1.0.0/original"

# Iterate over subdirectories in the base directory
for subdir in "$base_dir"/*; do
  # Skip if it's not a directory or if the name ends with .zip
  if [[ ! -d "$subdir" || "$subdir" == *.zip ]]; then
    continue
  fi

  # Get the subdirectory name
  collection_name=$(basename "$subdir")

  # Echo message indicating the directory being processed
  echo "Processing $collection_name..."

  # Special case for cqadupstack
  if [[ "$collection_name" == "cqadupstack" ]]; then
    # Iterate over the subdirectories inside cqadupstack
    for subsubdir in "$subdir"/*; do
      if [[ -d "$subsubdir" ]]; then
        subcollection_name=$(basename "$subsubdir")
        # Echo message for subdirectory inside cqadupstack
        echo "Processing $collection_name/$subcollection_name..."

        # Run the Python script in the background for each sub-subdirectory
        python scripts/bm25_benchmarking.py \
          --corpus-file "$subsubdir/corpus.jsonl" \
          --query-file "../anserini/tools/topics-and-qrels/topics.beir-v1.0.0-${collection_name}-${subcollection_name}.test.tsv.gz" \
          --output-file "runs/duckDB_bm25_beir_${collection_name}_${subcollection_name}.txt" >"logs/duckDB_bm25_beir_${collection_name}_${subcollection_name}.txt" &
      fi
    done
  else
    # Run the Python script in the background for other directories
    python scripts/bm25_benchmarking.py \
      --corpus-file "$subdir/corpus.jsonl" \
      --query-file "../anserini/tools/topics-and-qrels/topics.beir-v1.0.0-$collection_name.test.tsv.gz" \
      --output-file "runs/duckDB_bm25_beir_${collection_name}.txt" >logs/duckDB_bm25_beir_${collection_name}.txt &
  fi
done

# Wait for all background processes to finish
wait

# Echo message when all jobs are completed
echo "All directories processed."
