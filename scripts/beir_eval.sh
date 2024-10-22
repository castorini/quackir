#!/bin/bash

base_dir="/store/collections/beir-v1.0.0/original"
qrels_dir="/store/scratch/s42chen/anserini-tools/topics-and-qrels"
eval_dir="evals"

mkdir -p "$eval_dir"

for subdir in "$base_dir"/*; do
  if [[ ! -d "$subdir" || "$subdir" == *.zip ]]; then
    continue
  fi

  collection_name=$(basename "$subdir")
  echo "Evaluating $collection_name"
  if [[ "$collection_name" == "cqadupstack" ]]; then
    for subsubdir in "$subdir"/*; do
      if [[ -d "$subsubdir" ]]; then
        subcollection_name=$(basename "$subsubdir")
        run_file="runs/psql_fts_beir/cqadupstack_${subcollection_name}.txt"
        eval_file="$eval_dir/psql_fts_beir/cqadupstack_${subcollection_name}.txt"

        if [[ ! -f "$run_file" ]]; then
          echo "Run file does not exist for cqadupstack/$subcollection_name. Skipping..."
          continue
        fi

        qrels_file="$qrels_dir/qrels.beir-v1.0.0-$collection_name-$subcollection_name.test.txt"
        python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$qrels_file" "$run_file" >"$eval_file"
      fi
    done
  else
    run_file="runs/psql_fts_beir/${collection_name}.txt"
    eval_file="$eval_dir/psql_fts_beir/${collection_name}.txt"

    if [[ ! -f "$run_file" ]]; then
      echo "Run file does not exist for $collection_name. Skipping..."
      continue
    fi

    qrels_file="$qrels_dir/qrels.beir-v1.0.0-$collection_name.test.txt"
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$qrels_file" "$run_file" >"$eval_file"
  fi
done

echo "All collections processed."