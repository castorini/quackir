#
# QuackIR: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ._base import tokenize
from quackir._base import count_lines
import argparse
import json
import os
import gzip
from tqdm import tqdm 
    
def tokenize_tsv_file(input_file, open_cmd, num_lines):
    tokenized_data = []
    with open_cmd(input_file, 'rt') as f:
        for line in tqdm(f, total=num_lines, desc="Processing lines"):
            parts = line.strip().split('\t')
            id = parts[0]
            content = ' '.join(parts[1:])
            tokenized_data.append({id: tokenize(content)})    
    return tokenized_data

def tokenize_json_file(input_file, open_cmd, num_lines):
    tokenized_data = []
    with open_cmd(input_file, 'rt') as f:
        for line in tqdm(f, total=num_lines, desc="Processing lines"):
            obj = json.loads(line.strip())
            obj_items = list(obj.items())
            if not obj_items:
                continue
            _, id = obj_items[0]
            if 'title' in obj and 'text' in obj:
                content = f"{obj['title']} {obj['text']}"
            elif 'contents' in obj:
                content = obj['contents']
            else:
                content = ' '.join(str(v) for k, v in obj_items[1:])
            tokenized_data.append({id: tokenize(content)})
    return tokenized_data

def tokenize_file(filename):
    open_cmd = open
    if filename.endswith('.gz'):
        open_cmd = gzip.open
    num_lines = count_lines(filename, open_cmd)
    if '.jsonl' in filename:
        tokenized_data = tokenize_json_file(filename, open_cmd, num_lines)
    elif '.tsv' in filename:
        tokenized_data = tokenize_tsv_file(filename, open_cmd, num_lines)
    else:
        return []
    print(f"Tokenized {len(tokenized_data)} items from {filename}")
    return tokenized_data

def save_tokenized_data(tokenized_data, output_file):
    with open(output_file, 'w') as f:
        for item in tokenized_data:
            s = json.dumps({"id": list(item.keys())[0], "contents": list(item.values())[0]})
            f.write(s + '\n')
    print(f"Tokenized data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input file/directory containing text to tokenize.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output file where the tokenized text will be saved.")
    args = parser.parse_args()

    all_data = []
    if os.path.isdir(args.input):
        with os.scandir(args.input) as entries:
            for entry in entries:
                if entry.is_file():
                    all_data.extend(tokenize_file(entry.path))
    else:
        all_data = tokenize_file(args.input)
    save_tokenized_data(all_data, args.output)