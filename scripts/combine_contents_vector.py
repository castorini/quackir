import json
import argparse
import gzip
import os

def merge(parsed, embedding, output):
    with open(parsed, 'r') as f:
        parsed_contents = [json.loads(line) for line in f]

    with gzip.open(embedding, 'rt') as f:
        embedding_contents = [json.loads(line) for line in f]
        embedding_contents = {line["qid"]: line["vector"] for line in embedding_contents}

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as out:
        for i in range(len(parsed_contents)):
            row1 = parsed_contents[i]
            out.write(json.dumps({"id": row1['id'], "contents": row1['contents'], "vector": embedding_contents[row1['id']]}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-file", type=str, required=True)
    parser.add_argument("--embedding-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)

    args = parser.parse_args()
    
    merge(args.parsed_file, args.embedding_file, args.output_file)