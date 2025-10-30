# QuackIR: BM25 Baseline for NFCorpus with DuckDB

This guide contains instructions for running a BM25 baseline for NFCorpus using DuckDB.

If you're a Waterloo student traversing the [onboarding path](https://github.com/castorini/onboarding/blob/master/ura.md) (which [starts here](https://github.com/castorini/anserini/blob/master/docs/start-here.md)),
make sure you've first done the previous step, [a deeper dive into dense and sparse representations](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md).
In general, don't try to rush through this guide by just blindly copying and pasting commands into a shell;
that's what I call [cargo culting](https://en.wikipedia.org/wiki/Cargo_cult_programming).
Instead, really try to understand what's going on.

In this guide, we're going to go through the same type of retrieval (sparse lexical retrieval with BM25), but using a different tool: DuckDB instead of Lucene.
This is an important lesson: the conceptual framework (bi-encoder architecture) we've been learning applies across different implementations.

Why might you want to use DuckDB for retrieval?
Enterprises often already have relational databases deployed as part of their data infrastructure.
Rather than adding a separate search system (like Lucene) or vector database for retrieval-augmented generation (RAG) applications, organizations can leverage their existing databases.
This minimizes complexity in their software stack.

QuackIR demonstrates that relational database management systems (RDBMSes) like DuckDB can perform retrieval with effectiveness comparable to established IR toolkits.
In our examples of the bi-encoder framework for sparse retrieval, the document and query encoders generate sparse lexical representations, and retrieval is performed via BM25 scoring.
In Pyserini, this is handled by Lucene. Here, we'll use DuckDB's full-text search (FTS) capabilities instead.
Both approaches implement the same conceptual framework. They just use different underlying technologies.

**Learning outcomes** for this guide, building on previous steps in the onboarding path:

+ Be able to use QuackIR to index documents in NFCorpus with DuckDB and to build an FTS index.
+ Be able to use QuackIR to perform a batch retrieval run on queries from NFCorpus.
+ Be able to evaluate the retrieved results above.

## Installation

Make sure you have QuackIR installed through this [guide](https://github.com/castorini/quackir?tab=readme-ov-file#installation).

## Data Prep

In this lesson, we'll be working with [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/), a full-text learning to rank dataset for medical information retrieval.
The rationale is that the corpus is quite small &mdash; only 3633 documents &mdash; so everything here is practical to run on a laptop.

Let's first start by fetching the data:

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip -P collections
unzip collections/nfcorpus.zip -d collections
```

This just gives you an idea of what the corpus contains:

```bash
$ head -1 collections/nfcorpus/corpus.jsonl
{"_id": "MED-10", "title": "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland", "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of breast cancer death among statin users in a population-based cohort of breast cancer patients. The study cohort included all newly diagnosed breast cancer patients in Finland during 1995\u20132003 (31,236 cases), identified from the Finnish Cancer Registry. Information on statin use before and after the diagnosis was obtained from a national prescription database. We used the Cox proportional hazards regression method to estimate mortality among statin users with statin use as time-dependent variable. A total of 4,151 participants had used statins. During the median follow-up of 3.25 years after the diagnosis (range 0.08\u20139.0 years) 6,011 participants died, of which 3,619 (60.2%) was due to breast cancer. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic statin use were associated with lowered risk of breast cancer death (HR 0.46, 95% CI 0.38\u20130.55 and HR 0.54, 95% CI 0.44\u20130.67, respectively). The risk decrease by post-diagnostic statin use was likely affected by healthy adherer bias; that is, the greater likelihood of dying cancer patients to discontinue statin use as the association was not clearly dose-dependent and observed already at low-dose/short-term use. The dose- and time-dependence of the survival benefit among pre-diagnostic statin users suggests a possible causal effect that should be evaluated further in a clinical trial testing statins\u2019 effect on survival in breast cancer patients.", "metadata": {"url": "http://www.ncbi.nlm.nih.gov/pubmed/25329299"}}
```

We need to do a bit of data munging to merge the title and text fields into a single contents field.
QuackIR expects documents in the format `{"id": ..., "contents": ...}`.
This is similar to how Lucene indexer expects documents.

Run the following Python script:

```python
import json

with open("collections/nfcorpus/quackir_corpus.jsonl", "w") as out:
    with open("collections/nfcorpus/corpus.jsonl", "r") as f:
        for line in f:
            l = json.loads(line)
            s = json.dumps({"id": l["_id"], "contents": l["title"] + " " + l["text"]})
            out.write(s + "\n")
```

We also need to munge the queries into the right format (from jsonl to tsv).
You should be fine if you’ve followed the [BGE-base Baseline for NFCorpus](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md) setup in the onboarding path.
Otherwise, continue with the remaining steps.

Run the following Python script:

```python
import json

with open('collections/nfcorpus/queries.tsv', 'w') as out:
    with open('collections/nfcorpus/queries.jsonl', 'r') as f:
        for line in f:
            l = json.loads(line)
            out.write(l['_id'] + '\t' + l['text'] + '\n')
```

Similarly, we need to munge the relevance judgments (qrels) into the right format.
This command-line invocation does the trick:

```bash
tail -n +2 collections/nfcorpus/qrels/test.tsv | sed 's/\t/\tQ0\t/' > collections/nfcorpus/qrels/test.qrels
```

Okay, the data are ready now.

## Indexing

Before proceeding, make sure you are in your conda environment.

We can now index these documents using QuackIR:

```python
from quackir.index import DuckDBIndexer
from quackir import IndexType

table_name = "corpus"
index_type = IndexType.SPARSE
corpus_file = "collections/nfcorpus/quackir_corpus.jsonl"

indexer = DuckDBIndexer()
indexer.init_table(table_name, index_type)   # create table schema
indexer.load_table(table_name, corpus_file)  # load JSONL into DuckDB
indexer.fts_index(table_name)                # build FTS index over `contents`
indexer.close()
```

Let's break down what's happening here:

1. `init_table`: Creates a DuckDB table with the appropriate schema for storing documents. For sparse retrieval, this includes a text column for contents.
2. `load_table`: Inserts all documents from the JSONL file into the database table.
3. `fts_index`: Builds the full-text search (FTS) index.

We're using DuckDB's FTS extension, which implements BM25-style scoring similar to what you've seen with Lucene.

The above indexing command takes just a few seconds to run on a modern laptop, since we're simply loading documents into a database table and building an FTS index (no neural inference required, unlike dense retrieval models).

## Retrieval

We can now perform retrieval using QuackIR with the following Python code:

```python
from quackir.search import DuckDBSearcher
from quackir import SearchType
import csv
import pathlib

table_name = "corpus"
top_k = 1000

searcher = DuckDBSearcher()

with pathlib.Path("runs/run.quackir.duckdb.sparse.nfcorpus.txt").open("w") as out:
    with open("collections/nfcorpus/queries.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        for qid, qtext in r:
            hits = searcher.search(
                SearchType.SPARSE,
                query_string=qtext,
                table_names=[table_name],
                top_n=top_k,
            )
            for rank, h in enumerate(hits, start=1):
                docid = h[0]
                score = h[1]
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} QuackIR\n")

searcher.close()
```

Here, QuackIR uses DuckDB's FTS capabilities for BM25 scoring.
Under the hood, QuackIR translates your Python calls into SQL queries that DuckDB executes, but you don't need to write any SQL yourself.

The above retrieval command takes only a few minutes on a modern laptop since the corpus is small.
For larger corpora, you would see increased latency, which is one consideration when using RDBMSes for retrieval at scale.

## Single-Query Retrieval

While the batch run above processed all queries in a loop, let's now perform retrieval for an individual query so we can examine the results more closely and compare them to the batch output.

Here's the snippet of Python code that does what we want:

```python
from quackir.search import DuckDBSearcher
from quackir import SearchType

table_name = "corpus"
top_k = 10

searcher = DuckDBSearcher()
hits = searcher.search(
    SearchType.SPARSE,
    query_string="How to Help Prevent Abdominal Aortic Aneurysms",
    top_n=10,
    table_names=[table_name],
)

for i in range(0, top_k):
    print(f'{i+1:2} {hits[i][0]:7} {hits[i][1]:.6f}')

searcher.close()
```

The results should be as follows:

```
 1 MED-4555 9.790146
 2 MED-4423 6.976107
 3 MED-3180 5.932539
 4 MED-2718 4.941778
 5 MED-1309 4.792084
 6 MED-4424 4.714365
 7 MED-1705 4.596784
 8 MED-4902 4.412193
 9 MED-1009 4.314793
10 MED-1512 4.278235
```

We can verify these results against the batch run you performed above:

```bash
$ grep PLAIN-3074 runs/run.quackir.duckdb.sparse.nfcorpus.txt | head -10
PLAIN-3074 Q0 MED-4555 1 9.790146 QuackIR
PLAIN-3074 Q0 MED-4423 2 6.976107 QuackIR
PLAIN-3074 Q0 MED-3180 3 5.932539 QuackIR
PLAIN-3074 Q0 MED-2718 4 4.941778 QuackIR
PLAIN-3074 Q0 MED-1309 5 4.792084 QuackIR
PLAIN-3074 Q0 MED-4424 6 4.714365 QuackIR
PLAIN-3074 Q0 MED-1705 7 4.596784 QuackIR
PLAIN-3074 Q0 MED-4902 8 4.412193 QuackIR
PLAIN-3074 Q0 MED-1009 9 4.314793 QuackIR
PLAIN-3074 Q0 MED-1512 10 4.278235 QuackIR
```

Perfect!
The single-query results match the batch results exactly.

Additionally, notice how similar this QuackIR interface is to Pyserini's:

```python
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/lucene.nfcorpus')
hits = searcher.search('How to Help Prevent Abdominal Aortic Aneurysms')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.4f}')
```

Both provide a clean, Pythonic API for retrieval, even though they use different backends (DuckDB vs. Lucene).

## Evaluation

After the run finishes, we can evaluate the results using `trec_eval`:

```bash
python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
  runs/run.quackir.duckdb.sparse.nfcorpus.txt
```

The results will be something like:

```
ndcg_cut_10             all     0.3206
```

This nDCG@10 score of 0.3206 is very close to the Lucene baseline (0.3218 as you have seen in [this document](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md)), demonstrating that DuckDB achieves comparable effectiveness (nDCG@10) to established IR systems.

The small difference can be attributed to minor formula variations in BM25 implementation between DuckDB and Lucene.
Specifically, DuckDB's BM25 formula multiplies the score by (k1 + 1) and doesn't cache the document length metric, whereas Lucene's implementation differs slightly in these details.
Despite these implementation differences, the effectiveness is nearly identical, showing that the conceptual framework is what matters most.

If you've gotten here, congratulations!
You've completed your first indexing and retrieval run using DuckDB with QuackIR, and you've seen that relational databases can achieve competitive retrieval effectiveness.

And that's it!

## What Have We Learned?

To recap, what's the point of this exercise?

+ We've seen that BM25 retrieval is an instantiation of a bi-encoder architecture where the encoder representations are sparse lexical vectors.
+ For both DuckDB (via QuackIR) and Lucene (via Anserini & Pyserini), you now know how to build an index to store document vector representations.
+ For both DuckDB and Lucene, you now know how to encode a query into a sparse query vector.
+ For both DuckDB and Lucene, you know how to compute query-document scores via BM25 scoring.

More importantly, you've seen that DuckDB achieves effectiveness (nDCG@10 of 0.3206) very close to Lucene's baseline (0.3218).
This demonstrates that relational database management systems are viable for retrieval, especially for RAG applications.
For enterprises that already have relational databases deployed, they can add retrieval capabilities to their existing infrastructure without introducing new systems like Elasticsearch or dedicated vector databases.

Okay, that's it for this lesson. Before you move on, however, add an entry in the "[Reproduction Log](#reproduction-log)" at the bottom of this page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id that's on the main trunk of QuackIR, and use its 7-hexadecimal prefix for the link anchor text.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

+ Results reproduced by [@brandonzhou2002](https://github.com/brandonzhou2002) on 2025-10-30 (commit [`c9a80ed`](https://github.com/castorini/quackir/commit/c9a80edf993c3e7f17b34117d6f6b6e1e82051a6))