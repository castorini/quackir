# QuackIR: Usage of the Analysis API

QuackIR wraps Pyserini's default Lucene Analyzer to pre-process documents and queries. 
It can also munge data into the format required for indexing and search. 

## Python

#### tokenize

```python
def tokenize(to_tokenize: str) -> str:
```

Tokenizes given string with Pyserini's default Lucene Analyzer.
Joins the tokens with whitespace and returns it.

Used by `quackir.index` and `quackir.search` to tokenize documents and queries if `IndexType` or `SearchMethod` is `SPARSE` and the `--pretokenized` flag is not present. 

##### Parameters
+ `to_tokenize`: string to tokenize

## CLI

Using `quackir.analysis` directly tokenizes and munges the given input and save to the given output file. 

+ `--input` [Required]: 
Path to the input file/directory containing text to tokenize. 
Files can be compressed with `gzip`, and must be in either `jsonl` or `tsv` format. 
If the input is a directory, every file containing `.jsonl` or `.tsv` is processed.
Other files or subdirectories are skipped. 
If the file is in `jsonl`, the first field is taken as the identifier. 
If `title` and `text` are both fields, their values are concatenated and tokenized and other fields are ignored.
Otherwise, if `contents` is a field, its value is tokenized and other fields are ignored.
Otherwise, the values of all fields except the first are concatenated and processed. 
If the file is in `tsv`, the values of the first column is taken as the identifier. 
The values of all other columns are concatenated and processed. 
If a file is processed successfully, you should see a message with the file path and how many entries were processed from that file. 

+ `--output` [Required]:
Path to the output file where the tokenized text is saved.
This should be a `jsonl` file as the output is saved in `jsonl` format with the first field as `id`, and the second field as `contents`, the processed text. 
This format is exactly what `quackir.index` expects for sparse indexes and what `quackir.search` expects for sparse retrieval. 
An example output row:
`{"id": "PLAIN-68", "contents": "what actual chicken nugget"}`