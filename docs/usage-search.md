# QuackIR: Usage of the Search API

For sparse retrieval, QuackIR currently supports using BM25 in DuckDB and SQLite, and full text search in PostgreSQL.

For dense retrieval, QuackIR currently supports vector search in DuckDB and PostgreSQL. 

For hybrid retrieval, QuackIR currently supports reciprocal rank fusion in DuckDB and PostgreSQL with sparse and dense retrieval results. 

Using `quackir.search` directly searches the specified table of the specified database for the specified topics using the specified search method, and saves results to the specified output path. 

The appropriate database options must be provided.
For more details, see this [guide](./db-options.md). 

+ `--topics` [Required]: 
Path to the file containing queries in jsonl format.
Files can be compressed with `gzip`, and must be in `jsonl` format with the fields `id`, and the field `contents` if the `index-type` is `sparse` or the field `vector` if the `index-type` is `dense`. 
After the queries are successfully loaded, a message is printed with how many queries were loaded. 

+ `--output` [Required]: 
Path to save the search results. 
Results will be saved in the format of `doc_id Q0 query_id rank score run_tag` on each row.

+ `--search-method`:
Method of search to perform.
Available options: `sparse`, `dense`, `hybrid`.
If not provided, automatically infer from the column names in the index. 
That is, if `contents` is a column, `sparse` retreival is used. 
Otherwise, if `embedding` is a column, `dense` retrieval is used.
If two indexes are provided, check to ensure that one is `sparse` and one is `dense`, and `hybrid` retrieval is used.

+ `--index`:
Name of the table to search in. Accepts two values for hybrid search, one sparse and one dense; one value for sparse or dense search.If a table with the same name currently exists, it is dropped. 
Default is `corpus`.
Any dashes are changed to underscores for better compatibility. 
This cleaning also occurs during indexing.

+ `--pretokenized`:
Indicates if the queries are pretokenized. 
Default is `False`, meaning the queries will be tokenized during search.
Including this flag will turn off tokenizing during searching. 
Not considered for dense indexes. 

+ `--hits`:
Number of top results to return. 
Default is 1000.

+ `--rrf-k`: 
Parameter k needed for reciprocal rank fusion. 
Ignored for other search methods.
Default is 60.

+ `--run-tag`:
Tag to identify the run in the output file.
Default is the search method and database type joined by an underscore. 