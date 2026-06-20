# QuackIR: Dense Retrieval with Flock + Ollama

This short guide shows how to run a minimal dense-retrieval pipeline with the Flock extension and a local Ollama embedding model.
[Flock](https://github.com/dais-polymtl/flock) is an open-source DuckDB extension that integrates large language models and multimodal AI directly into SQL for tasks such as semantic search and retrieval-augmented generation (RAG).
It supports multiple providers, including Azure, OpenAI, and Ollama.
You can check out their [documentation](https://dais-polymtl.github.io/flock/docs/getting-started) for more details.
In this experiment, we use [Ollama](https://ollama.com), an open-source platform for running large language models locally on your own machine.

This guide assumes you’ve followed the NFCorpus setup in this [experiment](./experiments-nfcorpus.md).
If not, please follow that first.

## Install and start Ollama
+ Download and install Ollama from the [download page](https://ollama.com/download).
+ Ensure the service is running locally (default: `127.0.0.1:11434`). Start it with:

    ```bash
    OLLAMA_NUM_PARALLEL=2 \
    OLLAMA_MAX_QUEUE=2048 \
    ollama serve &
    ```

    The `ollama serve &` command starts the Ollama server in the background.
    The environment variables control concurrency and memory usage.
    If you encounter the `server busy, please try again. maximum pending requests exceeded` error while running the script, you can try the following solutions:

    1. Increase the `OLLAMA_MAX_QUEUE` value in the `ollama serve` command to allow more pending requests.
    2. Reduce the `batch_size` parameter in the `options_json` field when registering the model alias with `flock.create_model(...)` in [step 2 of the walkthrough](#full-walkthrough).

    Adjust these settings based on your hardware capacity and workload.

+ Download an embedding model from the Ollama model registry to run it locally on your machine. Visit the [Ollama model registry](https://ollama.com/search?c=embedding) to explore available embedding models.
In this guide, we use [`embeddinggemma`](https://ollama.com/library/embeddinggemma), developed by Google.

    ```bash
    ollama pull embeddinggemma
    ```

## Full walkthrough
Paste the following Python script into a file and execute it.
On a modern laptop with a CPU, the entire process usually completes in approximately 35 minutes.

```python
from quackir.flock import FlockManager
from quackir.index import DuckDBIndexer
from quackir.search import DuckDBSearcher
from quackir import IndexType
from pathlib import Path
import csv

"""
1) Configuration
"""
table_name = "corpus_dense"
corpus_file = "collections/nfcorpus/quackir_corpus.jsonl"
embedding_dim = 768
# This is the model name you pulled from Ollama in the previous setup
embedding_model = "embeddinggemma"
# Specify a unique alias for the model to be registered in Flock so that later calls from Flock can refer to it
model_alias = "Embedder"
# You should already have this file from the NFCorpus setup
queries_file = "collections/nfcorpus/queries.tsv"
output_path = Path(
    f"runs/run.quackir.duckdb.dense.flock.{embedding_model}.nfcorpus.txt"
)
top_k = 1000

"""
2) Register model alias
If you want to learn more about the Ollama setup in Flock, refer to the documentation at https://dais-polymtl.github.io/flock/docs/getting-started/ollama#ollama-setup).
"""
flock = FlockManager()
# This associates a model alias with the local Ollama embedding model
flock.create_model(
    alias=model_alias,
    provider_model=embedding_model,
    provider="ollama",
    options_json='{"tuple_format":"json", "batch_size":16}',
)

"""
3) Create dense table schema
"""
# Attach Flock to the DuckDB indexer
indexer = DuckDBIndexer(flock_manager=flock)
indexer.init_table(table_name, IndexType.DENSE, embedding_dim=embedding_dim)

"""
4) Load the corpus file and compute embeddings using Flock
"""
indexer.load_table(
    table_name,
    corpus_file,
    with_flock=True,  # invokes Flock to compute an embedding per row
    id_column="id",
    contents_column="contents",
    embedding_dim=embedding_dim,
)
indexer.close()

"""
5) Embed queries on-the-fly and write a TREC run file
"""
# Attach Flock to the DuckDB searcher
searcher = DuckDBSearcher(flock_manager=flock)

with output_path.open("w") as out, open(queries_file) as f:
    reader = csv.reader(f, delimiter="\t")
    for qid, qtext in reader:
        hits = searcher.embedding_search(
            query_embedding=qtext,  # raw query text; Flock generates the vector
            top_n=top_k,
            table_name=table_name,
            with_flock=True,  # enables Flock to compute the query embedding
            embedding_dim=embedding_dim,
        )
        for rank, (docid, score) in enumerate(hits, start=1):
            out.write(f"{qid} Q0 {docid} {rank} {score:.6f} QuackIR\n")

searcher.close()
```

## Evaluate with trec_eval

Adjust the run file path based on the embedding model you used. We use `embeddinggemma` in this example.

```bash
python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
    runs/run.quackir.duckdb.dense.flock.embeddinggemma.nfcorpus.txt
```

which should yield:

```
ndcg_cut_10             all     0.3562
```

if you are using `embeddinggemma`.

## Using Azure or OpenAI

To use Azure or OpenAI models instead of Ollama, you must provide your API key (and other parameters for Azure) when initializing `FlockManager`.
For implementation details and the full list of supported parameters, please refer to [`quackir/flock.py`](../quackir/flock.py).

For example, to use OpenAI's model, you can do the following:

```python
from quackir.flock import FlockManager
from quackir._base import SecretProvider

# Initialize FlockManager for OpenAI
flock = FlockManager(secret_type=SecretProvider.OPENAI, api_key="your-openai-api-key")
# Create a model alias for OpenAI's model
flock.create_model(
    alias="OpenAI-Embedder",
    provider="openai",
    provider_model="text-embedding-3-large",
    options_json='{"tuple_format":"json", "batch_size":16}',
)
```
