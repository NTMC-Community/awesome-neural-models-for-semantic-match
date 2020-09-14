# Ad-hoc Information Retrieval

---

**Information retrieval** (**IR**) is the activity of obtaining information resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing.  Here, the **Ad-hoc information retrieval** refer in particular to text-based retrieval where documents in the collection remain relative static and new queries are submitted to the system continually (cited from the [survey](https://arxiv.org/pdf/1903.06902.pdf)).

The number of queries is huge. Some benchmark datasets are listed in the
following,

## Classic Datasets

<table style="width: 600px; margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th align="left">Dataset</th>
      <th align="left">Genre</th>
      <th align="left">#Query</th>
      <th align="left">#Collections</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><a href="https://trec.nist.gov/data/t13_robust.html"><strong>Robust04</strong></a></td>
      <td align="left">news</td>
      <td align="left">250</td>
      <td align="left">0.5M</td>
    </tr>
    <tr>
      <td align="left"><a href="https://trec.nist.gov/data/webmain.html"><strong>ClueWeb09-Cat-B</strong></a></td>
      <td align="left">web</td>
      <td align="left">150</td>
      <td align="left">50M</td>
    </tr>
    <tr>
      <td align="left"><a href="https://trec.nist.gov/data/terabyte.html"><strong>Gov2</strong></a></td>
      <td align="left">.gov pages</td>
      <td align="left">150</td>
      <td align="left">25M</td>
    </tr>
    <tr>
      <td align="left"><a href="http://www.msmarco.org/dataset.aspx"><strong>MS MARCO (Document Ranking)</strong></a></td>
      <td align="left">web pages</td>
      <td align="left">367,013</td>
      <td align="left">3.2M</td>
    </tr>
    <tr>
      <td align="left"><a href="https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/"><strong>MQ2007</strong></a></td>
      <td align="left">.gov pages</td>
      <td align="left">1692</td>
      <td align="left">25M</td>
    </tr>
    <tr>
      <td align="left"><a href="https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/"><strong>MQ2008</strong></a></td>
      <td align="left">.gov pages</td>
      <td align="left">794</td>
      <td align="left">25M</td>
    </tr>
  </tbody>
</table>

* [**Robust04**](https://trec.nist.gov/data/t13_robust.html) is a small news dataset which contains about 0.5 million documents in total. The queries are collected from TREC Robust Track 2004. There are 250 queries in total.

* [**Cluebweb09**](https://trec.nist.gov/data/webmain.html) is a large Web collection which contains about 34 million documents in total. The queries are accumulated from TREC Web Tracks 2009, 2010, and 2011. There are 150 queries in total.

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection where the pages are crawled from .gov. It consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MS MARCO (Document Ranking)**](http://www.msmarco.org/dataset.aspx) provides a large number of information question-style queries from Bing's search logs. There passages are annotated by humans with relevant/non-relevant labels. There are 8,841822 documents in total. There are 808,731queries, 6,980 queries and 48,598 queries for train, validation and test, respectively.

* [**Million Query TREC 2007 (MQ2007)**](http://www.msmarco.org/dataset.aspx) is a LETOR benchmark dataset which uses Gov2 web collection. There are 1692 queries in MQ2007 with 65,323 labeled documents.

* [**Million Query TREC 2008 (MQ2008)**](http://www.msmarco.org/dataset.aspx) is another LETOR benchmark dataset which also uses Gov2 web collection. There are 784 queries in MQ2008 with 14,384 labeled documents.

## Neural Models
### Robust04

|  Model   | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py)  | [0.095](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)  | [0.171](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.201](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)|
| CDSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) |[0.067](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.125](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.146](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)  |
| ARC-I  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.041](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.065](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.066](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
| ARC-II  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.067](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.128](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.147](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
|  DRMM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/DRMM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py)| 0.279| 0.431 | 0.382 | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |
|  KNRM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/AdeDZY/K-NRM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/knrm.py) | — |[0.352](https://arxiv.org/pdf/1904.07094.pdf)|[0.409](https://arxiv.org/pdf/1904.07094.pdf) | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017](https://arxiv.org/pdf/1706.06613.pdf) |
|  CONV-KNRM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py)  | — | — |[0.416](https://arxiv.org/pdf/1905.09217.pdf)| [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) |
|  BERT-MaxP  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/AdeDZY/SIGIR19-BERT-IR) | — | — |0.469| [Deeper Text Understanding for IR with Contextual Neural Language Modeling, SIGIR 2019](https://arxiv.org/pdf/1905.09217.pdf) |
|  CEDR-DRMM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Georgetown-IR-Lab/cedr) | — | 0.459 |0.526| [CEDR: Contextualized Embeddings for Document Ranking, SIGIR 2019](https://arxiv.org/pdf/1904.07094.pdf) |
|  QINM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/TJUIRLAB/SIGIR20_QINM) | 0.294 | 0.408 |0.453| [A Quantum Interference Inspired Neural Matching Model for Ad-hoc Retrieval, SIGIR 2020](https://dl.acm.org/doi/pdf/10.1145/3397271.3401070) |


### ClueWeb09-B

|  Model   | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py)  | [0.054](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)  | [0.185](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.132](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)|
| CDSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) |[0.064](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.214](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.153](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)  |
| ARC-I  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.024](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.089](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.073](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
| ARC-II  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.033](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.123](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) | [0.087](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
|  DRMM  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/DRMM)[![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py)| 0.133| 0.365 | 0.258 | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |
|  CONV-KNRM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py)  | — | — |[0.270](https://arxiv.org/pdf/1905.09217.pdf)| [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) |
|  BERT-MaxP  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/AdeDZY/SIGIR19-BERT-IR) | — | — |0.289| [Deeper Text Understanding for IR with Contextual Neural Language Modeling, SIGIR 2019](https://arxiv.org/pdf/1905.09217.pdf) |
|  QINM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/TJUIRLAB/SIGIR20_QINM) | 0.134 | 0.375 |0.338| [A Quantum Interference Inspired Neural Matching Model for Ad-hoc Retrieval, SIGIR 2020](https://dl.acm.org/doi/pdf/10.1145/3397271.3401070) |

### MS MARCO (Document Ranking)

|  Model   | Code  | MRR@10 | nDCG@10 | Recall@10| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
| MatchPyramid| [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pl8787/MatchPyramid-TensorFlow) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/match_pyramid.py)| [0.286](https://arxiv.org/pdf/1710.05649.pdf)|[0.344](https://arxiv.org/pdf/2002.01854.pdf) |[0.531](https://arxiv.org/pdf/2002.01854.pdf) | [Text Matching as Image Recognition, AAAI 2016](https://arxiv.org/pdf/2002.01854.pdf) |
|  Duet  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py) |[0.266](https://arxiv.org/pdf/2002.01854.pdf) |[0.327](https://arxiv.org/pdf/2002.01854.pdf)|[0.520](https://arxiv.org/pdf/2002.01854.pdf)|[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) |
|  Co-PACRR  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/khui/copacrr) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master) | [0.284](https://arxiv.org/pdf/2002.01854.pdf) | [0.345](https://arxiv.org/pdf/2002.01854.pdf) | [0.543](https://arxiv.org/pdf/2002.01854.pdf) | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018](https://arxiv.org/pdf/1706.10192.pdf) |
|  KNRM  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/AdeDZY/K-NRM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/knrm.py) | [0.261](https://arxiv.org/pdf/2002.01854.pdf) |[0.323](https://arxiv.org/pdf/2002.01854.pdf)|[0.519](https://arxiv.org/pdf/2002.01854.pdf) | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017](https://arxiv.org/pdf/1706.06613.pdf) |
|  CONV-KNRM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py)  | [0.283](https://arxiv.org/pdf/2002.01854.pdf) | [0.345](https://arxiv.org/pdf/2002.01854.pdf) |[0.542](https://arxiv.org/pdf/2002.01854.pdf)| [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) |
|  BERT  | — | [0.352](https://arxiv.org/pdf/2002.01854.pdf) | [0.417](https://arxiv.org/pdf/2002.01854.pdf) |[0.623](https://arxiv.org/pdf/2002.01854.pdf)| [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019](https://arxiv.org/pdf/1905.09217.pdf) |
|  Transformer-Kernel  | — | 0.316 | 0.380 |0.586| [Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking, Arxiv 2020](https://arxiv.org/pdf/2002.01854.pdf) |

### MQ2007

|  Model   | Code  | MAP | P@10 | nDCG@10| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py)  | [0.409](https://arxiv.org/pdf/1805.05737.pdf)  | [0.352](https://arxiv.org/pdf/1805.05737.pdf) | [0.371](https://arxiv.org/pdf/1805.05737.pdf)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)|
| CDSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) |[0.364](https://arxiv.org/pdf/1710.05649.pdf)| [0.291](https://arxiv.org/pdf/1710.05649.pdf) |[0.325](https://arxiv.org/pdf/1710.05649.pdf)|[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)  |
| ARC-I  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.417](https://arxiv.org/pdf/1710.05649.pdf) | [0.364](https://arxiv.org/pdf/1710.05649.pdf) | [0.386](https://arxiv.org/pdf/1710.05649.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
| ARC-II  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.421](https://arxiv.org/pdf/1710.05649.pdf) | [0.366](https://arxiv.org/pdf/1710.05649.pdf) | [0.390](https://arxiv.org/pdf/1710.05649.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
|  DRMM  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/DRMM)[![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py)| [0.467](https://arxiv.org/pdf/1710.05649.pdf)| [0.388](https://arxiv.org/pdf/1805.05737.pdf) | [0.440](https://arxiv.org/pdf/1805.05737.pdf) | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |
| MatchPyramid| [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pl8787/MatchPyramid-TensorFlow) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/match_pyramid.py)| [0.434](https://arxiv.org/pdf/1710.05649.pdf)|[0.371](https://arxiv.org/pdf/1710.05649.pdf) |[0.409](https://arxiv.org/pdf/1710.05649.pdf) | [Text Matching as Image Recognition, AAAI 2016](https://arxiv.org/pdf/1602.06359.pdf) |
|  Duet  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py) |[0.474](https://arxiv.org/pdf/1805.05737.pdf) |[0.398](https://arxiv.org/pdf/1805.05737.pdf)|[0.453](https://arxiv.org/pdf/1805.05737.pdf)|[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) |
|  DeepRank  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pl8787/DeepRank_PyTorch) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master) |0.497 |0.412|0.482 | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017](https://arxiv.org/pdf/1710.05649.pdf) |
|  HiNT  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/HiNT) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master)|0.502 |0.447|0.490 |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018](https://arxiv.org/pdf/1805.05737.pdf)|

### MQ2008

|  Model   | Code  | MAP | P@10 | nDCG@10| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py)  | [0.391](https://arxiv.org/pdf/1805.05737.pdf)  | [0.221](https://arxiv.org/pdf/1805.05737.pdf) | [0.178](https://arxiv.org/pdf/1805.05737.pdf)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)|
| CDSSM  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py) |[0.395](https://arxiv.org/pdf/1710.05649.pdf)| [0.222](https://arxiv.org/pdf/1710.05649.pdf) |[0.175](https://arxiv.org/pdf/1710.05649.pdf)|[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)  |
| ARC-I  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.424](https://arxiv.org/pdf/1710.05649.pdf) | [0.311](https://arxiv.org/pdf/1710.05649.pdf) | [0.187](https://arxiv.org/pdf/1710.05649.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
| ARC-II  | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py) |[0.421](https://arxiv.org/pdf/1710.05649.pdf) | [0.229](https://arxiv.org/pdf/1710.05649.pdf) | [0.181](https://arxiv.org/pdf/1710.05649.pdf) |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)  |
|  DRMM  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/DRMM)[![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py)| [0.473](https://arxiv.org/pdf/1710.05649.pdf)| [0.245](https://arxiv.org/pdf/1805.05737.pdf) | [0.220](https://arxiv.org/pdf/1805.05737.pdf) | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) |
| MatchPyramid| [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pl8787/MatchPyramid-TensorFlow) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/match_pyramid.py)| [0.449](https://arxiv.org/pdf/1710.05649.pdf)|[0.239](https://arxiv.org/pdf/1710.05649.pdf) |[0.211](https://arxiv.org/pdf/1710.05649.pdf) | [Text Matching as Image Recognition, AAAI 2016](https://arxiv.org/pdf/1602.06359.pdf) |
|  Duet  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/bmitra-msft/NDRM/blob/master/notebooks/Duet.ipynb) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py) |[0.476](https://arxiv.org/pdf/1805.05737.pdf) |[0.240](https://arxiv.org/pdf/1805.05737.pdf)|[0.216](https://arxiv.org/pdf/1805.05737.pdf)|[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) |
|  DeepRank  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pl8787/DeepRank_PyTorch) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master) |0.498 |0.252|0.240 | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017](https://arxiv.org/pdf/1710.05649.pdf) |
|  HiNT  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/faneshion/HiNT)[![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master)|0.505 |0.255|0.244 |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018](https://arxiv.org/pdf/1805.05737.pdf)|
