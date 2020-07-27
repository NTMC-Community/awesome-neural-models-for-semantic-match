### Community Question Answer

---

**Community Question Answer** is to automatically search for relevant answers among many responses provided for a given question (Answer Selection), and search for relevant questions to reuse their existing answers (Question Retrieval).

Some benchmark datasets are listed in the following,

- [**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering by Microsoft Research.
- [**TRECQA**](https://trec.nist.gov/data/qa.html) dataset is created by [Wang et. al.](https://www.aclweb.org/anthology/D07-1003) from TREC QA track 8-13 data, with candidate answers automatically selected from each question’s document pool using a combination of overlapping non-stop word counts and pattern matching. This data set is one of the most widely used benchmarks for [answer sentence selection](https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)).
- [**SemEval-2015 Task 3**](http://alt.qcri.org/semeval2015/task3/) consists of two sub-tasks. In Subtask A, given a question (short title + extended description), and several community answers, classify each of the answer as definitely relevance (good), potentially useful (potential), or bad or irrelevant (bad, dialog, non-english other). In Subtask B, given a YES/NO question (short title + extended description), and a list of community answers, decide whether the global answer to the question should be yes, no, or unsure.
- [**SemEval-2016 Task 3**](http://alt.qcri.org/semeval2016/task3/) consists two sub-tasks, namely *Question-Comment Similarity* and *Question-Question Similarity*. In the *Question-Comment Similarity* task, given a question from a question-comment thread, rank the comments according to their relevance with respect to the question. In *Question-Question Similarity* task, given the new question, rerank all similar questions retrieved by a search engine.
- [**SemEval-2017 Task 3**](http://alt.qcri.org/semeval2017/task3/) contains two sub-tasks, namely *Question Similarity* and *Relevance Classification*. Given the new question and a set of related questions from the collection, the *Question Similarity* task is to rank the similar questions according to their similarity to the original question. While the *Relevance Classification* is to rank the answer posts according to their relevance with respect to the question based on a question-answer thread.

Representative neural matching models for CQA include:

**Waiting for perfection**

