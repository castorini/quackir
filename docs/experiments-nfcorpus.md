# QuackIR: Sparse and Dense Retrieval for NFCorpus with DuckDB

This guide contains instructions for running both sparse and dense retrieval baselines for NFCorpus using DuckDB.

If you're a Waterloo student traversing the [onboarding path](https://github.com/castorini/onboarding/blob/master/ura.md) (which [starts here](https://github.com/castorini/anserini/blob/master/docs/start-here.md)),
make sure you've first done the previous step, [a deeper dive into dense and sparse representations](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md).
In general, don't try to rush through this guide by just blindly copying and pasting commands into a shell;
that's what I call [cargo culting](https://en.wikipedia.org/wiki/Cargo_cult_programming).
Instead, really try to understand what's going on.

In this guide, we walk through both sparse and dense retrieval baselines, using DuckDB as the backend (instead of Lucene/Faiss).
The key lesson is that the same bi-encoder conceptual framework applies across implementations, regardless of the underlying engine.

Why might you want to use DuckDB for retrieval?
Enterprises often already have relational databases deployed as part of their data infrastructure.
Rather than adding a separate search system (like Lucene) or vector database for retrieval-augmented generation (RAG) applications, organizations can leverage their existing databases.
This minimizes complexity in their software stack.

QuackIR demonstrates that relational database management systems (RDBMSes) like DuckDB can perform retrieval with effectiveness comparable to established IR toolkits.

**Learning outcomes** for this guide, building on previous steps in the onboarding path:

+ Be able to index NFCorpus in DuckDB with QuackIR and build an FTS index for sparse retrieval.
+ Be able to encode documents and queries with the BGE-base model using Pyserini, producing L2-normalized 768‑d vectors.
+ Be able to compute query–document scores for dense retrieval (cosine similarity).
+ Be able to perform retrieval for both sparse and dense in DuckDB and write TREC-format run files.
+ Be able to evaluate runs with trec_eval (e.g., nDCG@10).

## Installation

Make sure you have QuackIR installed through the installation guide in the [README](../README.md#installation) file of this repository.

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

We need to do a bit of data munging to merge the `title` and `text` fields into a single contents field.
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
You should be fine if you’ve followed the [BGE-base Baseline for NFCorpus setup](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md#data-prep) in the onboarding path.
Otherwise, continue with the remaining steps in this section.

Run the following Python script:

```python
import json

with open("collections/nfcorpus/queries.tsv", "w") as out:
    with open("collections/nfcorpus/queries.jsonl", "r") as f:
        for line in f:
            l = json.loads(line)
            out.write(l["_id"] + "\t" + l["text"] + "\n")
```

Similarly, we need to munge the relevance judgments (qrels) into the right format.
This command-line invocation does the trick:

```bash
tail -n +2 collections/nfcorpus/qrels/test.tsv | sed 's/\t/\tQ0\t/' > collections/nfcorpus/qrels/test.qrels
```

The data is ready for now.
Later, for dense retrieval, you’ll need to encode the documents and queries with the BGE-base-en-v1.5 model using Pyserini.

## Sparse Retrieval with BM25

### Indexing

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

### Retrieval

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

### Single-Query Retrieval

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

searcher = LuceneSearcher("indexes/lucene.nfcorpus")
hits = searcher.search("How to Help Prevent Abdominal Aortic Aneurysms")

for i in range(0, 10):
    print(f"{i+1:2} {hits[i].docid:7} {hits[i].score:.4f}")
```

Both provide a clean, Pythonic API for retrieval, even though they use different backends (DuckDB vs. Lucene).

### Evaluation

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

This nDCG@10 score of 0.3206 is very close to the Lucene baseline (0.3218 as you have seen in this [guide](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md#sparse-retrieval-models)), demonstrating that DuckDB achieves comparable effectiveness (nDCG@10) to established IR systems.

The small difference can be attributed to minor formula variations in BM25 implementation between DuckDB and Lucene.
Specifically, DuckDB’s BM25 implementation differs from Lucene’s by explicitly including the (k1 + 1) multiplier in its scoring formula and by not employing a document-length caching strategy that Lucene uses.
Despite these implementation differences, the effectiveness is nearly identical, showing that the conceptual framework is what matters most.

## Dense Retrieval with BGE-base-en-v1.5

Now let's perform dense retrieval using the BGE-base-en-v1.5 model with DuckDB, similar to the sparse retrieval above and to the [BGE-base Baseline for NFCorpus](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md).

### Indexing

First, index the corpus using Pyserini (QuackIR does not include encoding functionality):

```bash
python -m pyserini.encode \
    input   --corpus collections/nfcorpus/corpus.jsonl \
                    --fields title text \
    output  --embeddings indexes/nfcorpus.bge-base-en-v1.5 \
    encoder --encoder BAAI/bge-base-en-v1.5 --l2-norm \
                    --device cpu \
                    --pooling mean \
                    --fields title text \
                    --batch 32
```

We're using the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) encoder, which can be found on HuggingFace.
Use `--device cuda` for a faster computation if you have a CUDA-enabled GPU.

The above indexing command takes a few minutes to run on a modern laptop, with most of the time occupied by performing neural inference using the CPU. Adjust the batch parameter above accordingly for your hardware.

Let's inspect the first line of the output:

```bash
head -n 1 indexes/nfcorpus.bge-base-en-v1.5/embeddings.jsonl
```

You should see a JSON line like:

```json
{"id": "MED-10", "contents": "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland\nRecent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of breast cancer death among statin users in a population-based cohort of breast cancer patients. The study cohort included all newly diagnosed breast cancer patients in Finland during 1995\u20132003 (31,236 cases), identified from the Finnish Cancer Registry. Information on statin use before and after the diagnosis was obtained from a national prescription database. We used the Cox proportional hazards regression method to estimate mortality among statin users with statin use as time-dependent variable. A total of 4,151 participants had used statins. During the median follow-up of 3.25 years after the diagnosis (range 0.08\u20139.0 years) 6,011 participants died, of which 3,619 (60.2%) was due to breast cancer. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic statin use were associated with lowered risk of breast cancer death (HR 0.46, 95% CI 0.38\u20130.55 and HR 0.54, 95% CI 0.44\u20130.67, respectively). The risk decrease by post-diagnostic statin use was likely affected by healthy adherer bias; that is, the greater likelihood of dying cancer patients to discontinue statin use as the association was not clearly dose-dependent and observed already at low-dose/short-term use. The dose- and time-dependence of the survival benefit among pre-diagnostic statin users suggests a possible causal effect that should be evaluated further in a clinical trial testing statins\u2019 effect on survival in breast cancer patients.", "vector": [0.008622120134532452, -0.06337106227874756, -0.02335186116397381, 0.028421154245734215, 0.03579297661781311, 0.007942342199385166, 0.02964012883603573, 0.00043221350642852485, -0.02297653816640377, -0.023847324773669243, 0.03230834752321243, 0.04067409038543701, -0.02417673170566559, 0.0138986986130476, 0.011943603865802288, 0.0647556483745575, 0.05228262394666672, 0.03968927636742592, -0.013745798729360104, 0.04104870930314064, 0.006505312863737345, -0.009682554751634598, 0.028484754264354706, 0.04309147968888283, 0.04389301687479019, 0.014030585065484047, -0.005556716118007898, -0.014659862965345383, -0.057544589042663574, 0.022318389266729355, 0.05189084634184837, -0.012022387236356735, 0.0019964759703725576, -0.004714031703770161, 0.03195904195308685, 0.02917233668267727, 0.016287868842482567, 0.03145579993724823, -0.011849591508507729, -0.011328321881592274, -0.08507925271987915, 0.001044508651830256, 0.009428598918020725, 0.014255858026444912, -0.007969190366566181, 0.0055594067089259624, 0.009332996793091297, 0.049483522772789, 0.025086522102355957, 0.01651439443230629, -0.042699236422777176, -0.007723322603851557, 0.027758974581956863, -0.007383323274552822, 0.020016731694340706, 0.053666673600673676, 0.02016506716609001, -0.016645483672618866, -0.0324278324842453, -0.04110582545399666, 0.026860477402806282, -0.020371077582240105, -0.008471024222671986, -0.0024583537597209215, 0.05535607784986496, 0.021897928789258003, 0.01514734048396349, 0.036236587911844254, -0.054827768355607986, -0.03218499943614006, -0.04786056652665138, -0.025619372725486755, 0.010108200833201408, -0.01951589062809944, -0.024307699874043465, -0.02532811649143696, -0.02532309666275978, 0.009710838086903095, 0.03479726240038872, 0.0076721603982150555, 0.04228603094816208, -0.05067875236272812, -0.05887739732861519, 0.0429401658475399, 0.007705802097916603, -0.021218236535787582, 0.011203198693692684, -0.03212164342403412, -0.05544498190283775, 0.05413612723350525, 0.019749276340007782, -0.006644509732723236, 0.05156448110938072, 0.04321654886007309, -0.015215126797556877, 0.0023013162426650524, 0.037672292441129684, -0.037330031394958496, -0.028630893677473068, -0.07159441709518433, -0.0258445143699646, -0.017856378108263016, -0.008025582879781723, -0.041975878179073334, -0.08309991657733917, -0.017599524930119514, 0.0008172528469003737, -0.011651149019598961, -0.002789416117593646, 0.01924474909901619, 0.0014990222407504916, 0.0032328166998922825, 0.0025373073294758797, 0.004709914326667786, -0.07884076237678528, 0.08686040341854095, 0.036614738404750824, 0.00574261462315917, 0.028026387095451355, -0.0255971010774374, 0.016149992123246193, -0.0020363598596304655, -0.027583453804254532, 0.08710300177335739, 0.010225361213088036, 0.05956534296274185, 0.046835388988256454, 0.02219289168715477, -0.0030234402511268854, -0.036180078983306885, 0.03670952841639519, 0.034989360719919205, -0.025044959038496017, 0.00784978549927473, -0.009957131929695606, 0.030129652470350266, 0.01189939584583044, -0.021507348865270615, 0.05394650995731354, -0.021068325266242027, 0.06327056139707565, -0.025946438312530518, 0.016497939825057983, -0.015332899987697601, 0.05472279340028763, -0.004697359167039394, 0.02616490051150322, -0.06709498912096024, -0.02677038311958313, 0.000530105666257441, 0.047110117971897125, 0.05287773534655571, 0.07214692234992981, -0.03445771709084511, 0.02105328068137169, 0.057197678834199905, 0.01609075255692005, -0.018671689555048943, -0.013978947885334492, -0.014514953829348087, 0.027872517704963684, -0.02138620801270008, 0.035791292786598206, -0.027879012748599052, 0.05059734359383583, 0.007713364437222481, 0.04979892075061798, 0.00852375477552414, -0.006739578675478697, -0.0029195642564445734, -0.07749541848897934, -0.031561702489852905, 0.03572840988636017, -0.01031755656003952, 0.05857040733098984, 0.046644680202007294, 0.040473438799381256, 0.004897954873740673, 0.016650555655360222, -0.04800570383667946, -0.07103344798088074, 0.06384459137916565, -0.011050250381231308, -0.02127092331647873, 0.026909228414297104, -0.0019792690873146057, 0.04276788607239723, -0.03043844923377037, 0.0228976272046566, 0.04548060521483421, -0.04200976714491844, -0.04938555881381035, 0.07612373679876328, -0.005026749335229397, 0.04424729198217392, -0.03086705505847931, -0.02326815016567707, 0.06466574221849442, -0.03404774144291878, -0.00989539921283722, -0.04411248490214348, -0.03958636894822121, 0.04677417874336243, -0.03163877874612808, -0.059881437569856644, 0.04037557169795036, -0.009368309751152992, -0.050923727452754974, -0.013585731387138367, 0.06327008455991745, 0.0057418616488575935, 0.012160010635852814, 0.021894119679927826, -0.012562060728669167, 0.021096929907798767, 0.024631621316075325, 0.018034515902400017, 0.033586494624614716, 0.0208132341504097, -0.04138242080807686, 0.046011775732040405, -0.017006361857056618, -0.01201860886067152, 0.014662418514490128, 0.027377426624298096, 0.1167626678943634, 0.05981298163533211, -0.03971121087670326, -0.025903379544615746, 0.035634394735097885, -0.035835206508636475, 0.016570333391427994, 0.013552743010222912, 0.0343184731900692, 0.03998902440071106, -0.013632485643029213, -0.0019644610583782196, 0.004803300369530916, 0.03364669904112816, -0.06026051566004753, 0.010749098844826221, 0.036723654717206955, -0.0013309981441125274, 0.04042435437440872, -0.006625865586102009, -0.0025365762412548065, -0.04427317902445793, -0.025715187191963196, -0.07852515578269958, -0.03927928954362869, -0.035133861005306244, 0.0102605652064085, 0.01983293704688549, 0.04734630137681961, -0.0265625212341547, -0.030234547331929207, -0.018536312505602837, 0.021659886464476585, 0.09783004224300385, 0.017209401354193687, -0.03857705369591713, 0.017818402498960495, 0.012420549057424068, -0.04303627461194992, -0.03555299714207649, -0.055933933705091476, -0.0511956624686718, -0.025124968960881233, 0.04151231795549393, -0.02421090006828308, 0.04811754450201988, 0.016138728708028793, -0.031581416726112366, -0.006835354026407003, -0.04255136474967003, 0.021432604640722275, -0.010275091044604778, 0.031730152666568756, -0.023477235808968544, -0.004066327586770058, 0.038994111120700836, 0.07687394320964813, 0.014628740027546883, 0.0029643524903804064, -0.014926797710359097, -0.009285174310207367, -0.0054087392054498196, -0.059660084545612335, -0.014070646837353706, 0.019349154084920883, 0.003446436021476984, 0.04797764867544174, -0.04434001073241234, -0.026474596932530403, 0.08829451352357864, -0.0034198244102299213, -0.02053118869662285, 0.039768315851688385, -0.0003170176059938967, -0.007635214366018772, 0.03079577162861824, -0.01052887737751007, 0.04133693128824234, -0.023540597409009933, -0.0033574742265045643, 0.01527608186006546, -0.009754469618201256, -0.003916330635547638, -0.2517063021659851, 0.01838449202477932, -0.05794893205165863, -0.026056576520204544, 0.044736459851264954, -0.005873083136975765, 0.010994892567396164, -0.003958981949836016, -0.02293390780687332, -0.006762501318007708, 0.051671694964170456, 0.027491403743624687, 0.00038986981962807477, 0.03680797293782234, 0.05910812318325043, 0.0257424283772707, -0.005066074430942535, -0.04977826401591301, -0.024508455768227577, 0.027619875967502594, 0.06391693651676178, -0.024187572300434113, -0.030041882768273354, 0.04377374425530434, -0.00017797363398130983, 0.06524733453989029, -0.0053156460635364056, 0.008734399452805519, -0.06627269834280014, -0.008077690377831459, 0.0012328341836109757, -0.009791869670152664, 0.004965405911207199, 0.0007891810382716358, -0.006176920607686043, -0.024232879281044006, 0.0343281514942646, 0.025804396718740463, -0.017428310588002205, -0.02922564558684826, -0.005153389181941748, -0.02735402062535286, -0.054890308529138565, -0.03003263846039772, 0.04466429352760315, 0.0048899841494858265, -0.01906997710466385, 0.0038779100868850946, -0.015158580616116524, 0.06485128402709961, 0.05417424067854881, -0.03287767246365547, -0.0947900041937828, 0.026602575555443764, -0.015003223903477192, 0.0020439636427909136, -0.028206411749124527, 0.022354668006300926, 0.0017371721332892776, 0.018181603401899338, -0.032929155975580215, -0.01699761301279068, 0.02074216865003109, -0.06841344386339188, -0.02336084097623825, -0.04484669864177704, -0.030281290411949158, -0.08896780014038086, 0.051691342145204544, 0.05005740001797676, -0.015084107406437397, -0.010679986327886581, -0.01767134480178356, -0.07174547016620636, -0.009524279274046421, -0.01807907596230507, -0.031447865068912506, -0.011525922454893589, -0.017568429931998253, 0.05164175108075142, -0.005842228420078754, -0.0350932702422142, 0.004937856923788786, 0.012908450327813625, 0.0295337475836277, -0.013810992240905762, -0.0008648047805763781, 0.008422845043241978, -0.03471039980649948, -0.024595191702246666, 0.03116689622402191, -0.014575343579053879, 0.013427729718387127, 0.0029896798077970743, 0.007597525604069233, 0.06995032727718353, -0.024731166660785675, -0.0004971280577592552, -0.027285698801279068, -0.014523890800774097, 0.039536863565444946, -0.06466805189847946, -0.011019422672688961, -0.10616614669561386, 0.0006664090906269848, -0.008417644537985325, -0.05169380083680153, -0.038327381014823914, -0.0015683032106608152, -0.03763023763895035, 0.04807626083493233, -0.015474632382392883, 0.04198138043284416, -0.07852376252412796, 0.05109952390193939, -0.022260745987296104, 0.02237023413181305, 0.021475132554769516, 0.052048102021217346, -0.012464353814721107, -0.01946188509464264, 0.041814740747213364, -0.04307717829942703, -0.06848210096359253, -0.003766514826565981, -0.03202012926340103, 0.0029086219146847725, -0.020148666575551033, -0.031045343726873398, 0.05254058167338371, -0.003155439393594861, 0.006031625904142857, 0.04079952463507652, 0.016926711425185204, 0.031572699546813965, 0.017160115763545036, -0.04442627355456352, -0.008212612941861153, -0.01460721530020237, -0.021369269117712975, -0.013950115069746971, -0.01815885119140148, 0.002156701870262623, 0.00897601805627346, 0.021867554634809494, -0.002872703829780221, -0.009588095359504223, 0.04355943948030472, 0.004650390241295099, 0.016333650797605515, -0.0008567959303036332, 0.010504764504730701, 0.01026679202914238, -0.03362880274653435, 0.02014055848121643, -0.011330398730933666, 0.07002225518226624, 0.001606709323823452, -0.028321512043476105, -0.0981077179312706, -0.01882287673652172, -0.03076987713575363, -0.02437923476099968, -0.07594127953052521, -0.019227853044867516, 0.03756847232580185, -0.031411733478307724, -0.034580253064632416, 0.020907480269670486, -0.004885967820882797, 0.023827362805604935, -0.05050009861588478, -0.001186753623187542, 0.04103769361972809, -0.041257135570049286, 0.004582696128636599, -0.039925120770931244, -0.021326670423150063, 0.0025736403185874224, -0.0006123656057752669, -0.007588579319417477, -0.007302815094590187, -0.0350869819521904, -0.0018840244738385081, 0.06304729729890823, -0.0009570205584168434, 0.002325434470549226, -0.02878226898610592, -0.04192982614040375, 0.010296674445271492, -0.0016512551810592413, 0.012718449346721172, -0.026572300121188164, -0.02040625922381878, -0.0409117192029953, -0.017340054735541344, 0.008053508587181568, 0.037768442183732986, -0.009076321497559547, 0.027109984308481216, -0.0037614149041473866, -0.0022489784751087427, -0.019440513104200363, -0.017970696091651917, 0.025333184748888016, -0.07073900103569031, 0.055850252509117126, 0.036459781229496, -0.030680114403367043, 0.038222406059503555, -0.034615203738212585, -0.020133020356297493, -0.01001423317939043, -0.03287705034017563, 0.018969031050801277, -0.02863348461687565, -0.010127446614205837, -0.013348260894417763, -0.011270706541836262, -0.005394072737544775, 0.027535714209079742, -0.04006905108690262, -0.01626807264983654, -0.02977793477475643, -0.027774237096309662, 0.07977882772684097, -0.03628141060471535, -0.02567904070019722, 0.06237189099192619, -0.030603274703025818, -0.007519636303186417, -0.04863238334655762, -0.01697596348822117, -0.00534774037078023, -0.011556151323020458, 0.001513472874648869, -0.029492083936929703, 0.047637734562158585, -0.007698755711317062, 0.05532432347536087, -0.039365384727716446, 0.011324110440909863, -0.012202040292322636, 0.006554356310516596, 0.0038749196100980043, 0.016996964812278748, -0.004990784917026758, -0.009869441390037537, -0.0030675758607685566, 0.028904415667057037, 0.009001107886433601, -0.03132912144064903, -0.038449835032224655, -0.07209984213113785, 0.031196229159832, -0.0162802767008543, -0.04803360253572464, -0.014436044730246067, -0.04762427136301994, 0.06926500797271729, 0.015563977882266045, -0.0036423285491764545, -0.023088663816452026, 0.05162425339221954, 0.039612945169210434, 0.031225087121129036, -0.02375541441142559, -0.010525454767048359, 0.01294933632016182, -0.04432980716228485, 0.010536769405007362, -0.06009084731340408, -0.005689569283276796, -0.0026579732075333595, -0.006593944504857063, 0.04569216072559357, 0.006276915315538645, -0.021852390840649605, 0.00662132166326046, -0.023391900584101677, -0.05452138930559158, -0.015706844627857208, -0.02659899927675724, -0.0220958199352026, 0.025784241035580635, -0.041344840079545975, -0.0010155766503885388, 0.020701725035905838, -0.03616702929139137, 0.013446932658553123, 0.02174183540046215, 0.05053505674004555, -0.01396771240979433, -0.013504568487405777, -0.030498605221509933, -0.004904359579086304, 0.016448641195893288, 0.03835294768214226, 0.009328953921794891, 0.019314562901854515, -0.04942171648144722, 0.03967202827334404, 0.028642117977142334, 0.01782502979040146, -0.0184226855635643, 0.011970870196819305, 0.03495662659406662, -0.04643202945590019, 0.008577186614274979, 0.06245844438672066, -0.021795885637402534, -0.05965516343712807, 0.09974049031734467, 0.033688973635435104, -0.08254394680261612, -0.04446493834257126, -0.023280849680304527, -0.0430467315018177, 0.024019742384552956, 0.009301076643168926, 0.04508829116821289, -9.957757720258087e-05, 0.052547913044691086, -0.012622137553989887, 0.03567618876695633, 0.053956419229507446, -0.009584004990756512, 0.009265209548175335, 0.04131435602903366, 0.08182474225759506, 0.06447847932577133, 0.01103825680911541, 0.009560646489262581, 0.08370484411716461, -0.026280246675014496, -0.05695081129670143, -0.002005340065807104, -0.06613634526729584, -0.03860766813158989, 0.0001323385804425925, 0.05951641499996185, 0.009064849466085434, 0.026858776807785034, 0.05112046003341675, 0.042533326894044876, 0.013623698614537716, -0.03437633439898491, 0.021362006664276123, 0.037340197712183, -0.03184260427951813, 0.022333744913339615, 0.022926803678274155, -0.03841143101453781, -0.01341285090893507, -0.01920386590063572, 0.02026844397187233, 0.02093750424683094, -0.027859684079885483, -0.005441716872155666, -0.023506203666329384, -0.003110450692474842, -0.028841061517596245, 0.06967300921678543, -0.017844008281826973, 0.008084907196462154, 0.006785358767956495, 0.04449838027358055, 0.025090264156460762, 0.060202911496162415, 0.03568999841809273, 0.041407033801078796, -0.03147143870592117, -0.008315715938806534, -0.029807455837726593, -0.01924814097583294, -0.02225893922150135, 0.01407643686980009, -0.07912902534008026, 0.005634678993374109, 0.0628199353814125, -0.039888445287942886, -0.045268408954143524, -0.03360116481781006, -0.008149943314492702, -0.035492390394210815, -0.08286912739276886, -0.003226440865546465, 0.018036946654319763, -0.0281694233417511, -0.007029657252132893, -0.00628931587561965, 0.02862301468849182, 0.05475284531712532, 0.012088162824511528, -0.026693983003497124, -0.029441896826028824, 0.0674603208899498, -0.01075498852878809, 0.0406336709856987, 0.03623943403363228, 0.014890574850142002, -0.014492830261588097, -0.033103663474321365, 0.06978529691696167, -0.012052707374095917, 0.01583988033235073, -0.026158347725868225, 0.007792163174599409, -0.04483083635568619, 0.020206157118082047, 0.01717263273894787, -0.0064927441999316216, -0.04652602970600128, 0.038319386541843414, -0.01543544139713049, 0.014177619479596615, 0.047803521156311035, 0.0219427440315485, -0.016023442149162292, -0.022356880828738213, -0.0285554900765419, -0.039934612810611725, 0.02454284019768238, 0.04741397127509117, -0.012934278696775436, 0.056905876845121384, -0.007141695357859135, -0.02069825306534767, 0.01715407334268093, 0.0004099213401786983, -0.007892165333032608, -0.017549991607666016, -0.02600794844329357, -0.047048419713974, -0.06607965379953384, -0.07582640647888184, 0.0070698750205338, -0.006839000154286623, -0.05022914707660675, -0.020212125033140182, -0.01177893951535225, 0.01789718121290207, 0.0313078798353672, -0.03547031432390213, 0.022546790540218353, -0.01630554348230362, -0.03834671154618263, -0.06902875751256943, -0.05771346017718315, 0.006877224892377853, -0.03809134662151337, -0.051259443163871765, 0.030589357018470764, -0.031152060255408287, 0.062243275344371796, -0.025698939338326454, 0.010695773176848888, -0.0031933877617120743, -0.03759193792939186, -0.016199057921767235]}
```

We also need to encode the queries:

```bash
python -m pyserini.encode \
    input   --corpus collections/nfcorpus/queries.jsonl \
                    --fields text \
    output  --embeddings collections/nfcorpus/queries.bge-base-en-v1.5 \
    encoder --encoder BAAI/bge-base-en-v1.5 --l2-norm \
                    --device cpu \
                    --pooling mean \
                    --batch 32
```

Then convert the Pyserini-generated JSONL file into a Parquet file containing only the `id` and `vector` fields.
QuackIR’s DuckDB indexer expects exactly these two fields (it does not use `contents`).
Alternatively, you can keep the data in JSONL format instead of converting it to Parquet, as long as you include only the two required fields.
The example here simply demonstrates that the system can read a Parquet file. You can learn more about the advantages of the Parquet format if you're interested.

```python
import json
import pyarrow as pa
import pyarrow.parquet as pq

data = {"id": [], "vector": []}

with open("indexes/nfcorpus.bge-base-en-v1.5/embeddings.jsonl", "r") as f_in:
    for line in f_in:
        doc = json.loads(line)
        data["id"].append(doc["id"])
        data["vector"].append(doc["vector"])

table = pa.table(data)
pq.write_table(table, "indexes/nfcorpus.bge-base-en-v1.5/embeddings.parquet")
```

Now we can load the pre-encoded vectors into DuckDB:

```python
from quackir.index import DuckDBIndexer
from quackir import IndexType

table_name = "corpus_dense"
index_type = IndexType.DENSE
corpus_embeddings = "indexes/nfcorpus.bge-base-en-v1.5/embeddings.parquet"

indexer = DuckDBIndexer()
indexer.init_table(table_name, index_type, embedding_dim=768)
indexer.load_table(table_name, corpus_embeddings)

indexer.close()
```

Here's what happens in the code above:

1. `init_table`: Creates a DuckDB table with columns for document ID and embedding vector (768 dimensions for BGE-base).
2. `load_table`: Reads the Parquet files containing the pre-encoded vectors and inserts them into the table.

Notice there is no `fts_index` step. Dense retrieval uses vector similarity instead of BM25.

This step is very fast (a few seconds) since it only loads precomputed vectors into the database.

### Retrieval

You can now perform dense retrieval with the encoded queries:

```python
from quackir.search import DuckDBSearcher
from quackir import SearchType
import json
import pathlib

table_name = "corpus_dense"
top_k = 1000

searcher = DuckDBSearcher()

with pathlib.Path("runs/run.quackir.duckdb.dense.nfcorpus.txt").open("w") as out:
    with open("collections/nfcorpus/queries.bge-base-en-v1.5/embeddings.jsonl") as f:
        for line in f:
            query = json.loads(line)
            qid = query["id"]
            qvector = query["vector"]

            hits = searcher.search(
                SearchType.DENSE,
                query_embedding=qvector,
                table_names=[table_name],
                top_n=top_k,
            )

            for rank, h in enumerate(hits, start=1):
                docid = h[0]
                score = h[1]
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} QuackIR\n")

searcher.close()
```

Here, QuackIR uses DuckDB's [`array_cosine_similarity`](https://duckdb.org/docs/stable/sql/functions/array#array_cosine_similarityarray1-array2) function for vector similarity computation. Under the hood, it performs an exact search by computing the cosine similarity between the query vector and all document vectors.

Mathematical Note: Because we encode with `--l2-norm`, embeddings are unit vectors. Cosine similarity then equals dot product: cos(θ) = (𝐮 · 𝐯) / (‖𝐮‖‖𝐯‖) ⇒ if ‖𝐮‖ = ‖𝐯‖ = 1, then cos(θ) = 𝐮 · 𝐯.

This retrieval takes a few minutes since the corpus is small. We omit a single-query example here because the batch run is simply a loop over single queries, and the results are identical, as you have seen in sparse retrieval above.

### Evaluation

After the run finishes, evaluate the results using `trec_eval`:

```bash
python -m pyserini.eval.trec_eval \
    -c -m ndcg_cut.10 collections/nfcorpus/qrels/test.qrels \
    runs/run.quackir.duckdb.dense.nfcorpus.txt
```

You should see results like the following:

```
ndcg_cut_10             all     0.3808
```

This matches the Pyserini/Faiss baseline (nDCG@10 of 0.3808), as reported [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-nfcorpus.md#evaluation).

If you've gotten here, congratulations!
You've completed sparse and dense retrieval runs using DuckDB with QuackIR, and you've seen that relational databases can achieve competitive retrieval effectiveness.

And that's it!

## What Have We Learned?

To recap, what's the point of this exercise?

+ We see that sparse retrieval and dense retrieval are both instantiations of a bi-encoder architecture; the only difference is the encoder representations (sparse lexical vectors vs. dense vectors).
+ With DuckDB, you build an FTS index for sparse and load pre‑encoded embeddings for dense.
+ For both a sparse retrieval model and a dense retrieval model, you know how to compute query–document scores. In practice with DuckDB, we use BM25 for sparse and cosine similarity (equals dot product under L2‑normalization) for dense.
+ Finally, for both sparse and dense, you can perform retrieval “by hand”: iterate over all DuckDB‑stored document vectors and compute query–document scores in a brute‑force manner.

More importantly, you can see that DuckDB achieves an effectiveness (nDCG@10 = 0.3206) very close to Lucene’s baseline (0.3218) for the sparse model, and exactly the same effectiveness (nDCG@10 = 0.3808) as the Faiss baseline for the dense model.
This demonstrates that relational database management systems are viable for retrieval, especially for RAG applications.
For enterprises that already have relational databases deployed, they can add retrieval capabilities to their existing infrastructure without introducing new systems like Elasticsearch or dedicated vector databases.

Okay, that's it for this lesson. Before you move on, however, add an entry in the [Reproduction Log](#reproduction-log) at the bottom of this page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id that's on the main trunk of QuackIR, and use its 7-hexadecimal prefix for the link anchor text.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

+ Results reproduced by [@brandonzhou2002](https://github.com/brandonzhou2002) on 2025-10-30 (commit [`c9a80ed`](https://github.com/castorini/quackir/commit/c9a80edf993c3e7f17b34117d6f6b6e1e82051a6))