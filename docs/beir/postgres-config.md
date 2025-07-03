# QuackIR: PostgreSQL Full Text Search Configuration

The current PostgreSQL sparse retrieval configuration in QuackIR uses the "simple" configuration with no stopwords.
This is consistent with our implementation of retrieval in DuckDB and SQLite where we tokenize with Pyserini's Lucene analyzer and turn of the default text-processing of the RDBMS as much as possible.

The current PostgreSQL sparse retrieval method is as follows:

```python
def fts_search(self, query_string, top_n=5, table_name="corpus"):
    ts_query = self.clean_tsquery(query_string)
    query = f"""
    SELECT 
        id, 
        ts_rank(to_tsvector('simple', contents), to_tsquery('simple', %s)) AS score
    FROM 
        {table_name}
    WHERE
        to_tsvector('simple', contents) @@ to_tsquery('simple', %s)
    ORDER BY 
        score DESC
    LIMIT %s
    """
    cur = self.conn.cursor()
    cur.execute(query, (ts_query, ts_query, top_n))
    return cur.fetchall()
```

The source code of which can be found [here](../../quackir/search/_postgres.py).

However, since PostgreSQL does not implement BM25 like DuckDB and SQLite, this is not necessarily a fair comparison.
We can obtain better results with a modified configuration using the "english" configuration, which uses the Snowball stemmer for English, and length normalization by dividing a document's score by 1 + the logarithm of its length. 
However, the latency increases compared to the "simple" configuration and takes more than one second per query on even the smallest of the BEIR datasets.

You can find more information about the English configuration [here](https://www.postgresql.org/docs/current/textsearch-dictionaries.html) and length normalization [here](https://www.postgresql.org/docs/current/textsearch-controls.html).

The sparse retrieval method with the modified configuration would look like:

```python
def fts_search(self, query_string, top_n=5, table_name="corpus"):
    ts_query = self.clean_tsquery(query_string)
    query = f"""
    SELECT 
        id, 
        ts_rank(to_tsvector('english', contents), to_tsquery('english', %s), 1) AS score
    FROM 
        {table_name}
    WHERE
        to_tsvector('english', contents) @@ to_tsquery('english', %s)
    ORDER BY 
        score DESC
    LIMIT %s
    """
    cur = self.conn.cursor()
    cur.execute(query, (ts_query, ts_query, top_n))
    return cur.fetchall()
```

The retrieval effectiveness of the two configurations measured in nDCG@10 is shown below, where "Default" refers to the "simple" configuration with no stopwords and "Modified" refers to the "English" configuration with length normalization:

| Corpus                    | Default | Modified |
| ------------------------- | ------- | -------- |
| `trec-covid`              | \-      | \-       |
| `bioasq`                  | \-      | \-       |
| `nfcorpus`                | 0.2965  | 0.3055   |
| `nq`                      | \-      | \-       |
| `hotpotqa`                | \-      | \-       |
| `fiqa`                    | 0.0918  | 0.1805   |
| `signal1m`                | \-      | \-       |
| `trec-news`               | \-      | \-       |
| `robust04`                | \-      | \-       |
| `arguana`                 | 0.0690  | 0.2549   |
| `webis-touche2020`        | \-      | \-       |
| `cqadupstack-android`     | 0.2607  | 0.3423   |
| `cqadupstack-english`     | 0.2252  | 0.2378   |
| `cqadupstack-gaming`      | 0.3436  | 0.4116   |
| `cqadupstack-gis`         | 0.1864  | 0.2621   |
| `cqadupstack-mathematica` | 0.1215  | 0.1847   |
| `cqadupstack-physics`     | 0.2053  | 0.3003   |
| `cqadupstack-programmers` | 0.1866  | 0.2503   |
| `cqadupstack-stats`       | 0.1828  | 0.2588   |
| `cqadupstack-tex`         | 0.1303  | 0.2111   |
| `cqadupstack-unix`        | 0.1678  | 0.2546   |
| `cqadupstack-webmasters`  | 0.2319  | 0.2865   |
| `cqadupstack-wordpress`   | 0.1280  | 0.2405   |
| `quora`                   | \-      | \-       |
| `dbpedia-entity`          | \-      | \-       |
| `scidocs`                 | 0.0907  | 0.1165   |
| `fever`                   | \-      | \-       |
| `climate-fever`           | \-      | \-       |
| `scifact`                 | 0.5692  | 0.6064   |