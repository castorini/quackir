# QuackIR: Usage of the Index API

For sparse indexes, QuackIR currently supports BM25 indexing in DuckDB and SQLite, and full text search indexing in PostgreSQL.

For dense indexes, QuackIR currently supports vector indexes DuckDB and PostgreSQL.

Using `quackir.index` directly loads the specified input into the specified table for the specified index type in the specified database, tokenizing the contents if the index is sparse unless otherwise indicated. 

The appropriate database options must be provided.
For more details, see this [guide](./db-options.md). 

+ `--input` [Required]: 
Path to the file or folder containing data to index.
Files must be in either `jsonl` or `parquet` format. 
If the input is a directory, every file ending in `.jsonl` or `.parquet` is processed.
Other files or subdirectories are skipped. 
If the file is in `jsonl`, it is expected that it has the fields `id`, and the field `contents` if the `index-type` is `sparse` or the field `vector` if the `index-type` is `dense`. 
If the file is in `parquet`, it is expected that the `index-type` is `dense` and that there are two columns in the file, the first being the id and the second being the vector with the key `vector`. 
It follows that if the file is in `parquet` the `db-type` cannot be `SQLITE`.
After every successfully processed file, a message is printed with how many entries are currently in the index. 

+ `--index-type` [Required]:
Type of index to create.
Available options: `sparse`, `dense`.
If the index-type is `sparse`, a table with the columns `id` and `contents` is created; and the input files cannot be in `parquet`.
If the index-type is `dense`, a table with the columns `id` and `embedding` is created; and the `db-type` cannot be `SQLITE`.

+ `--index`:
Name of the table to create. 
Default is `corpus`. 
If a table with the same name currently exists, it is dropped. 
Any dashes are changed to underscores for better compatibility. 
This cleaning also occurs during search.

+ `--pretokenized`:
Indicates if the contents are pretokenized. 
Default is `False`, meaning the contents will be tokenized during indexing.
Including this flag will turn off tokenizing during indexing. 
Not considered for dense indexes. 

+ `--dimension`:
Dimension of the embedding vector. 
Default is 768. 
Not considered for sparse indexes. 