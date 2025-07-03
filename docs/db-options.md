# QuackIR: Database Options for Indexing and Search

QuackIR currently supports DuckDB, SQLite, and PostgreSQL. 

For more information on how to set up the databases, see [these guides](./db_guides/).

For `quackir.index` and `quackir.search`, one of the three above database types must be indicated with the appropriate configs.

+ `--db-type` [Required]:
Type of database to use.
Available options: `duckdb`, `sqlite`, `postgres`.
Dotenv key: `DB_TYPE`

+ `--db-path`: 
Path to the database file used for DuckDB and SQLite. 
Ignored for Postgres.
Default is `database.db`.
Dotenv key: `DB_PATH`.

+ `--db-name`: 
Name of the database for Postgres. 
Ignored for DuckDB and SQLite.
Default is `quackir`.
Dotenv key: `DB_NAME`. 

+ `--db-user`:
Username for Postgres. 
Ignored for DuckDB and SQLite.
Default is `postgres`.
Dotenv key: `DB_USER`.

Note that if you intend to use both DuckDB and SQLite, point `--db-path` to different paths. 

Dotenv configuration will override command line arguments. 