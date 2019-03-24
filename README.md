<div align="center">
	<img width="300" src="artworks/awaresome.svg" alt="Awesome">
	<br>
	<br>
	<p><b>Awaresome Neural Models for Semantic Match</b></p>
</div>
<br>
<p align="center">
    <sub>A collection of papers maintained by MatchZoo Team.</sub>
    <br>
    <sub>Checkout our open source toolkit <a href="https://github.com/faneshion/MatchZoo">MatchZoo</a> for more information!</sub>
</p>
<br>

Text matching is a core component in many natural language processing tasks, where many task can be viewed as a matching between two texts input.

​							$$\text{Match}(s, t) = g(f(\psi(s), \phi(t)))​$$

Where $s​$ and $t​$ are source text input and target text input, respectively. The $\psi​$ and $\phi​$ are representation function for input $s​$ and $t​$, respectively. The $f​$ is the interaction function, and $g​$ is the aggregation function. The representative matching tasks are as follows:

<table>
  <tr>
    <th width=30%, bgcolor=#999999 >Tasks</th> 
    <th width=20%, bgcolor=#999999>Source Text</th>
    <th width="20%", bgcolor=#999999>Target Text</th>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Ad-hoc Information Retrieval </td>
    <td align="center", bgcolor=#eeeeee> query </td>
    <td align="center", bgcolor=#eeeeee> document (title/content) </td>
  </tr>
    <tr>
    <td align="center", bgcolor=#eeeeee> Community Question Answer </td>
    <td align="center", bgcolor=#eeeeee> question </td>
    <td align="center", bgcolor=#eeeeee> question/answer </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Paraphrase Indentification </td>
    <td align="center", bgcolor=#eeeeee> string 1 </td>
    <td align="center", bgcolor=#eeeeee> string 2 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Natural Language Inference </td>
    <td align="center", bgcolor=#eeeeee> premise </td>
    <td align="center", bgcolor=#eeeeee> hypothesis </td>
  </tr>
</table>



### Ad-hoc Information Retrieval

---

**Information retrieval** (**IR**) is the activity of obtaining information system resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing.  Here, the **Ad-hoc information retrieval** refer in particular to text-based retrieval where the number of queries is huge. Some benchmark datasets are listed in the following,

* [**Robust04**](https://trec.nist.gov/data/t13_robust.html) is a small news dataset which contains about 0.5 million documents in total. The queries are collected from TREC Robust Track 2004. There are 250 queries in total.

* [**Cluebweb09**](https://trec.nist.gov/data/webmain.html) is a large Web collection which contains about 34 million documents in total. The queries are accumulated from TREC Web Tracks 2009, 2010, and 2011. There are 150 queries in total.

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection where the pages are crawled from .gov. It consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MSMARCO Passage Reranking**](http://www.msmarco.org/dataset.aspx) provides a large number of information question-style queries from Bing's search logs. There passages are annotated by humans with relevant/non-relevant labels. There are 8,841822 documents in total. There are 6,980 queries and 48,598 queries for validation and test, respectively.

Neural information retrieval (NeuIR) models include:

1. **DSSM** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py) [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) ![status](artworks/ready.svg)

   Learning Deep Structured Semantic Models for Web Search using Clickthrough Data. *CIKM 2013*.
   [website](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) | [tutorial](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/dl-summer-school-2017.-Jianfeng-Gao.v2.pdf)

2. **CDSSM**  [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf) ![status](artworks/ready.svg)

   Learning Semantic Representations Using Convolutional Neural Networks for Web Search. *WWW 2014*.
   [website](https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)

   A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. *CIKM 2014*.
   [website](https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf)

3. **DRMM**  [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py) [[pdf]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) ![status](artworks/ready.svg)

   A Deep Relevance Matching Model for Ad-hoc Retrieval. *DRMM 2016*.
   [website](https://arxiv.org/abs/1711.08611) | [paper](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)

4. **KNRM**  [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/knrm.py) [[pdf]](https://arxiv.org/pdf/1706.06613.pdf) ![status](artworks/ready.svg)

   End-to-End Neural Ad-hoc Ranking with Kernel Pooling. *SIGIR 2017*
   [website](https://arxiv.org/abs/1706.06613) | [paper](https://arxiv.org/pdf/1706.06613.pdf)

5. **CONV-KNRM**  [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py) [[pdf]](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search. *WSDM 2018*
   [website](https://dl.acm.org/citation.cfm?id=3159659) | [paper](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)

6. #### **Duet** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py) [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) ![status](artworks/ready.svg)

   Learning to Match using Local and Distributed Representations of Text for Web Search. *WWW 2017*
   [website](https://dl.acm.org/citation.cfm?id=3052579) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf)

7. **Co-PACRR**  [[code]]() [[pdf]](https://arxiv.org/pdf/1706.10192.pdf) ![status](artworks/not-in-plan.svg)

   Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval. *WSDM 2018*.
   [website](https://arxiv.org/abs/1706.10192) | [paper](https://arxiv.org/pdf/1706.10192.pdf)

8. **LSTM-RNN**  [[code not ready]]() [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf) ![status](artworks/not-in-plan.svg)

   Deep Sentence Embsedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval. *TASLP 2016*.
   [website](https://arxiv.org/abs/1502.06922) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf)

9. **DRMM_TKS** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm_tks.py) [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2) ![status](artworks/ready.svg)

   A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM). *CCIR 2018*.
   [website](https://arxiv.org/abs/1711.08611) | [paper](https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2)

10. **DeepRank**  [[code]]() [[pdf]](https://arxiv.org/pdf/1710.05649.pdf) ![status](artworks/progress.svg)

    DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval. *CIKM 2017*
    [website](https://arxiv.org/abs/1710.05649) | [paper](https://arxiv.org/pdf/1710.05649.pdf)

11. **HiNT**  [[code]]() [[pdf]](https://arxiv.org/pdf/1805.05737.pdf) ![status](artworks/progress.svg)

    Modeling Diverse Relevance Patterns in Ad-hoc Retrieval. *SIGIR 2018*.
    [website](https://arxiv.org/abs/1805.05737) | [paper](https://arxiv.org/pdf/1805.05737.pdf)

12. **…**

    

### Paraphrase Identification

---

**Paraphrase Identification** is an task to determine whether two sentences have the same meaning, a problem considered a touchstone of natural language understanding.

Some benchmark datasets are listed in the following,

- [**MSRP**](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2Fdefault.aspx) is short for Microsoft Research Paraphrase Corpus. It contains 5,800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship.
- [**Quora Question Pairs**](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) is a task released by Quora which aims to identify duplicate questions. It consists of over 400,000 pairs of questions on Quora, and each question pair is annotated with a binary value indicating whether the two questions are paraphrase of each other.

A list of neural matching models for paraphrase identification models are as follows,

1. **ARC-I** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) [[pdf]](https://arxiv.org/pdf/1503.03244.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Network Architectures for Matching Natural Language Sentences. *NIPS 2014*.
   [website](https://arxiv.org/abs/1503.03244) | [paper](https://arxiv.org/pdf/1503.03244.pdf)

2. **ARC-II** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arcii.py) [[pdf]](https://arxiv.org/pdf/1503.03244.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Network Architectures for Matching Natural Language Sentences. *NIPS 2014*.
   [website](https://arxiv.org/abs/1503.03244) | [paper](https://arxiv.org/pdf/1503.03244.pdf)

3. **MV-LSTM** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/mvlstm.py) [[pdf]](https://arxiv.org/pdf/1511.08277.pdf) ![status](artworks/ready.svg)

   A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations. *AAAI 2016*.
   [website](https://arxiv.org/abs/1511.08277) | [paper](https://arxiv.org/pdf/1511.08277.pdf)

4. **MatchPyramid** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/match_pyramid.py) [[pdf]](https://arxiv.org/pdf/1602.06359.pdf) ![status](artworks/ready.svg)

   Text Matching as Image Recognition. *AAAI 2016*.
   [website](https://arxiv.org/abs/1602.06359) | [paper](https://arxiv.org/pdf/1602.06359.pdf)

5. **Match-SRNN**  [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/matchsrnn.py) [[pdf]](https://arxiv.org/pdf/1604.04378.pdf) ![status](artworks/progress.svg)

   Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN. *IJCAI 2016*.
   [website](https://arxiv.org/abs/1604.04378) | [paper](https://arxiv.org/pdf/1604.04378.pdf)

6. **MultiGranCNN** [[code]]() [[pdf]](https://aclanthology.info/pdf/P/P15/P15-1007.pdf) ![status](artworks/not-in-plan.svg)

   MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity. *ACL 2015*.
   [website](https://arxiv.org/abs/1604.04378) | [paper](https://aclanthology.info/pdf/P/P15/P15-1007.pdf)

7. **ABCNN** [[code]]() [[pdf]](https://arxiv.org/pdf/1512.05193.pdf) ![status](artworks/not-in-plan.svg)

   ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs. *ACL 2016*.
   [website](https://arxiv.org/abs/1512.05193) | [paper](https://arxiv.org/pdf/1512.05193)

8. **MwAN** [[code]]() [[pdf]](https://www.ijcai.org/proceedings/2018/0613.pdf) ![status](artworks/progress.svg)

   Multiway Attention Networks for Modeling Sentences Pairs. *IJCAI 2018*.

   [website](https://www.ijcai.org/proceedings/2018/0613.pdf) | [paper](https://www.ijcai.org/proceedings/2018/0613.pdf)

9. **…**

### Community Question Answer

---

**Community Question Answer** is to automatically search for relevant answers among many responses provided for a given question (Answer Selection), and search for relevant questions to reuse their existing answers (Question Retrieval).

Some benchmark datasets are listed in the following,

- [**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering by Microsoft Research.
- [**SemEval-2015 Task 3**](http://alt.qcri.org/semeval2015/task3/) consists of two sub-tasks. In Subtask A, given a question (short title + extended description), and several community answers, classify each of the answer as definitely relevance (good), potentially useful (potential), or bad or irrelevant (bad, dialog, non-english other). In Subtask B, given a YES/NO question (short title + extended description), and a list of community answers, decide whether the global answer to the question should be yes, no, or unsure.
- [**SemEval-2016 Task 3**](http://alt.qcri.org/semeval2016/task3/) consists two sub-tasks, namely *Question-Comment Similarity* and *Question-Question Similarity*. In the *Question-Comment Similarity* task, given a question from a question-comment thread, rank the comments according to their relevance with respect to the question. In *Question-Question Similarity* task, given the new question, rerank all similar questions retrieved by a search engine.
- [**SemEval-2017 Task 3**](http://alt.qcri.org/semeval2017/task3/) contains two sub-tasks, namely *Question Similarity* and *Relevance Classification*. Given the new question and a set of related questions from the collection, the *Question Similarity* task is to rank the similar questions according to their similarity to the original question. While the *Relevance Classification* is to rank the answer posts according to their relevance with respect to the question based on a question-answer thread.

Representative neural matching models for CQA include:

1. **aNMM** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/anmm.py) [[pdf]]() ![status](artworks/ready.svg)

   aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model. *CIKM 2016*.
   [website](https://arxiv.org/abs/1801.01641) | [paper](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240)

2. **MCAN** [[code]]() [[pdf]]() ![status](artworks/not-in-plan.svg)

   Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction. *KDD 2018*.
   [website](https://arxiv.org/abs/1806.00778) | [paper](https://arxiv.org/pdf/1806.00778.pdf)

3. **MIX** [[code]]() [[pdf]](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/hchen-kdd18.pdf) ![status](artworks/not-in-plan.svg)

   MIX: Multi-Channel Information Crossing for Text Matching. *KDD 2018*.
   [website](http://www.kdd.org/kdd2018/accepted-papers/view/mix-multi-channel-information-crossing-for-text-matching) | [paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/hchen-kdd18.pdf)

4. **…**

   



### Natural Language Inference

---

**Natural Language Inference** is the task of determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise".

It is worth to note that most models designed for Paraphrase Identification task can also be applied on NLI tasks, such as MatchPyramid, Match-SRNN, MV-LSTM, MwAN, and MultiGranCNN.

Some benchmark datasets are listed in the following,

- [**SNLI**](https://nlp.stanford.edu/projects/snli/) is the short of Stanford Natural Language Inference, which has 570k human annotated sentence pairs. Thre premise data is draw from the captions of the Flickr30k corpus, and the hypothesis data is manually composed.
- [**MultiNLI**](https://www.nyu.edu/projects/bowman/multinli/) is short of Multi-Genre NLI, which has 433k sentence pairs, whose collection process and task detail are modeled closely to SNLI. The premise data is collected from maximally broad range of genre of American English such as non-fiction genres (SLATE, OUP, GOVERNMENT, VERBATIM, TRAVEL), spoken genres (TELEPHONE, FACE-TO-FACE), less formal written genres (FICTION, LETTERS) and a specialized one for 9/11.

Representative neural matching models for NLI include:

1. **Match-LSTM** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/match_lstm.py) [[pdf]]() ![status](artworks/ready.svg)

   Learning Natural Language Inference with LSTM. *NAACL 2016*.
   [website](https://arxiv.org/abs/1512.08849) | [paper](http://www.aclweb.org/anthology/N16-1170)

2. **BiMPM** [[code]](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/bimpm.py) [[pdf]]() ![status](artworks/ready.svg)

   Bilateral Multi-Perspective Matching for Natural Language Sentences. *IJCAI 2017*.
   [website](https://arxiv.org/abs/1702.03814) | [paper](https://arxiv.org/pdf/1702.03814.pdf)

3. **ESIM** [[code]]() [[pdf]]() ![status](artworks/progress.svg)

   Enhanced LSTM for Natural Language Inference. *ACL 2017*.
   [website](https://arxiv.org/abs/1609.06038) | [paper](https://arxiv.org/pdf/1609.06038.pdf)

4. **DIIN** [[code]]() [[pdf]](https://arxiv.org/pdf/1709.04348.pdf) ![status](artworks/progress.svg)

   Natural Lanuguage Inference Over Interaction Space. *ICLR 2018*.

   [website](https://arxiv.org/pdf/1709.04348.pdf) | [paper](https://arxiv.org/pdf/1709.04348.pdf)

5. **…**


### Healthcheck

```python
pip3 install -r requirements.txt
python3 healthcheck.py
```