# QuackIR

[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

QuackIR is a toolkit for reproducible information retrieval research with relational database management systems. 
Sparse retrieval is available with DuckDB, SQLite, and PostgreSQL. 
Dense and hybrid retrieval are available with DuckDB and PostgreSQL.
Analysis with the porter tokenizer is provided via wrapping Pyserini's Lucene analyzer.

## Installation

### Clone Repository

```bash
git clone https://github.com/castorini/quackir.git --recurse-submodules
```

### Install Dependencies

```bash
conda create -n quackir python=3.10
conda activate quackir
conda install -c conda-forge postgresql pgvector openjdk=21 maven -y
cd quackir
pip install -r requirements.txt
```

### Initialize PostgreSQL 

```bash
initdb -D mydb
pg_ctl -D mydb -l logfile start &
createdb quackir
psql quackir
create user postgres superuser;
create extension vector;
\q
```

## Quick Start

To create a sparse index with DuckDB:

```python
from quackir.index import DuckDBIndexer
from quackir import IndexType

table_name = "corpus"
index_type = IndexType.SPARSE
corpus_file = YOUR_CORPUS_FILE_PATH

indexer = DuckDBIndexer()
indexer.init_table(table_name, index_type)
indexer.load_table(table_name, corpus_file)
indexer.fts_index(table_name)

indexer.close()
```

To perform sparse retrieval: 

```python
from quackir.search import DuckDBSearcher
from quackir import SearchType

table_name = "corpus"
query = "what is a lobster roll"
search_type = SearchType.SPARSE

searcher = DuckDBSearcher()
results = searcher.search(
    search_type, query_string=query, table_names=[table_name]
)
print(results)

searcher.close()
```

For using commands, see the [documentation](./docs/).

## Reproduce

For step-by-step reproduction of BEIR experiments, see these [docs](./docs/beir/).

To reproduce all BEIR experiments, run the following command and find the results in [logs](./logs/):

```bash
bash ./scripts/beir/run.sh
```