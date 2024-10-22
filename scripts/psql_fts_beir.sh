#!/bin/bash

base_dir="/store/collections/beir-v1.0.0/original"
topics_qrels_dir="/store/scratch/s42chen/anserini-tools/topics-and-qrels"
corpus_tsv_dir="collections"
runs_dir="runs/psql_fts_beir"
logs_dir="logs"

db_user="postgres"
db_password="YOUR_PASSWORD_HERE"
db_host="localhost"
db_port=5433
db_name="beir_datasets"

PGPASSWORD=$db_password psql -h $db_host -p $db_port -U $db_user -d postgres -c "CREATE DATABASE $db_name;" 2>/dev/null

prepare_query_file() {
    local collection=$1
    local subcollection=$2
    local query_file_base="${topics_qrels_dir}/topics.beir-v1.0.0-${collection}${subcollection:+-$subcollection}.test"
    local query_file_tsv="${query_file_base}.tsv"
    local query_file_gz="${query_file_base}.tsv.gz"

    if [[ -f "$query_file_tsv" ]]; then
        echo "$query_file_tsv"
    elif [[ -f "$query_file_gz" ]]; then
        echo "Decompressing $query_file_gz..."
        gunzip -c "$query_file_gz" > "$query_file_tsv"
        echo "$query_file_tsv"
    else
        echo "Error: Neither $query_file_tsv nor $query_file_gz exists." >&2
        return 1
    fi
}

run_psql_fts_searcher() {
    local corpus=$1
    local query=$2
    local output=$3
    local log=$4
    local table_prefix=$5

    python scripts/psql_fts_searcher.py \
        --corpus "$corpus" \
        --query "$query" \
        --output "$output" \
        --db-name "$db_name" \
        --db-user "$db_user" \
        --db-password "$db_password" \
        --db-host "$db_host" \
        --db-port "$db_port" \
        --table-prefix "$table_prefix" \
        >"$log"
}

get_or_create_tsv_corpus() {
    local jsonl_corpus=$1
    local dataset_name=$2
    local tsv_corpus="$corpus_tsv_dir/${dataset_name}.tsv"

    if [[ -f "$tsv_corpus" ]]; then
        echo "$tsv_corpus"
    else
        python scripts/jsonl_to_tsv.py --input_file "$jsonl_corpus" -o "$tsv_corpus"
        echo "$tsv_corpus"
    fi
}

mkdir -p $corpus_tsv_dir $runs_dir $logs_dir

for subdir in "$base_dir"/*; do
    if [[ ! -d "$subdir" || "$subdir" == *.zip ]]; then
        continue
    fi

    collection_name=$(basename "$subdir")

    echo "Processing $collection_name..."

    if [[ "$collection_name" == "cqadupstack" ]]; then
        for subsubdir in "$subdir"/*; do
            if [[ -d "$subsubdir" ]]; then
                subcollection_name=$(basename "$subsubdir")
                echo "Processing $collection_name/$subcollection_name..."

                output_file="$runs_dir/${collection_name}_${subcollection_name}.txt"
                if [[ -f "$output_file" ]]; then
                    echo "Skipping $collection_name/$subcollection_name as $output_file already exists."
                    continue
                fi

                query_file=$(prepare_query_file "$collection_name" "$subcollection_name")
                if [[ $? -ne 0 ]]; then
                    echo "Skipping $collection_name/$subcollection_name due to missing query file."
                    continue
                fi

                corpus_file=$(get_or_create_tsv_corpus "$subsubdir/corpus.jsonl" "${collection_name}_${subcollection_name}")
                table_prefix="${collection_name}_${subcollection_name}"
                run_psql_fts_searcher \
                    "$corpus_file" \
                    "$query_file" \
                    "$output_file" \
                    "$logs_dir/psql_fts_beir_${collection_name}_${subcollection_name}.txt" \
                    "$table_prefix"
            fi
        done
    else
        output_file="$runs_dir/${collection_name}.txt"
        if [[ -f "$output_file" ]]; then
            echo "Skipping $collection_name as $output_file already exists."
            continue
        fi

        query_file=$(prepare_query_file "$collection_name")
        if [[ $? -ne 0 ]]; then
            echo "Skipping $collection_name due to missing query file."
            continue
        fi

        corpus_file=$(get_or_create_tsv_corpus "$subdir/corpus.jsonl" "${collection_name}")
        table_prefix="${collection_name}"
        run_psql_fts_searcher \
            "$corpus_file" \
            "$query_file" \
            "$output_file" \
            "$logs_dir/psql_fts_beir_${collection_name}.txt" \
            "$table_prefix"
    fi
done

echo "All directories processed."