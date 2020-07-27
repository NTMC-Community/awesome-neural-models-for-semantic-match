### Ad-hoc Information Retrieval

---

**Information retrieval** (**IR**) is the activity of obtaining information system resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing.  Here, the **Ad-hoc information retrieval** refer in particular to text-based retrieval where documents in the collection remain relative static and new queries are submitted to the system continually (cited from the [survey](https://arxiv.org/pdf/1903.06902.pdf)).

the number of queries is huge. Some benchmark datasets are listed in the following,

* [**Robust04**](https://trec.nist.gov/data/t13_robust.html) is a small news dataset which contains about 0.5 million documents in total. The queries are collected from TREC Robust Track 2004. There are 250 queries in total.

* [**Cluebweb09**](https://trec.nist.gov/data/webmain.html) is a large Web collection which contains about 34 million documents in total. The queries are accumulated from TREC Web Tracks 2009, 2010, and 2011. There are 150 queries in total.

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection where the pages are crawled from .gov. It consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MSMARCO Passage Reranking**](http://www.msmarco.org/dataset.aspx) provides a large number of information question-style queries from Bing's search logs. There passages are annotated by humans with relevant/non-relevant labels. There are 8,841822 documents in total. There are 808,731queries, 6,980 queries and 48,598 queries for train, validation and test, respectively.

Neural information retrieval (NeuIR) models include:

**Waiting for perfection. **