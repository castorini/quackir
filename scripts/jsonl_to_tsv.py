import json
import csv
import argparse
import os
from tqdm import tqdm

def convert_jsonl_to_tsv(input_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.tsv'
    
    if os.path.exists(output_file):
        return output_file

    total_lines = sum(1 for _ in open(input_file, 'r'))
    with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
        tsv_writer.writerow(['id', 'contents'])  # Write header
        for line in tqdm(jsonl_file, total=total_lines, desc=f"Converting {os.path.basename(input_file)} to TSV", unit="lines"):
            try:
                data = json.loads(line)
                text = data.get('text', '').replace('\n', ' ').replace('\r', '')
                text = text.replace('\\', '\\\\')
                text = text.replace('"', '\\"')
                tsv_writer.writerow([data.get('_id', ''), text])
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Skipping line due to missing key: {e}")
                print(f"Problematic line: {line.strip()}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL file to TSV format.")
    parser.add_argument("--input_file", help="Path to the input JSONL file")
    parser.add_argument("-o", "--output_file", help="Path to the output TSV file (optional)")
    args = parser.parse_args()

    convert_jsonl_to_tsv(args.input_file, args.output_file)