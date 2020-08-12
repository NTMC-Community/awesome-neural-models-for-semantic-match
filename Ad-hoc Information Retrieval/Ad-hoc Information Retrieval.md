# Ad-hoc Information Retrieval

---

**Information retrieval** (**IR**) is the activity of obtaining information system resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing.  Here, the **Ad-hoc information retrieval** refer in particular to text-based retrieval where documents in the collection remain relative static and new queries are submitted to the system continually (cited from the [survey](https://arxiv.org/pdf/1903.06902.pdf)).

the number of queries is huge. Some benchmark datasets are listed in the following,

## Classic Datasets

|  Dataset   | Genre  | #Query | #Collections |
|  ----  | ----  |----  | ----  |
|  Robust04  | news  | 250  | 0.5M  |
|  ClueWeb09-Cat-B  |  web | 150  | 50M  |
|  Gov2  |  .gov pages | 150  | 25M  |
|  MS MARCO |  web pages | 367,013 | 3.2M  |
|  MQ2007 |  .gov pages | 1692  | 25M  |
|  MQ2008 |  .gov pages | 794 | 25M  |

* [**Robust04**](https://trec.nist.gov/data/t13_robust.html) is a small news dataset which contains about 0.5 million documents in total. The queries are collected from TREC Robust Track 2004. There are 250 queries in total.

* [**Cluebweb09**](https://trec.nist.gov/data/webmain.html) is a large Web collection which contains about 34 million documents in total. The queries are accumulated from TREC Web Tracks 2009, 2010, and 2011. There are 150 queries in total.

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection where the pages are crawled from .gov. It consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MSMARCO Passage Reranking**](http://www.msmarco.org/dataset.aspx) provides a large number of information question-style queries from Bing's search logs. There passages are annotated by humans with relevant/non-relevant labels. There are 8,841822 documents in total. There are 808,731queries, 6,980 queries and 48,598 queries for train, validation and test, respectively.

## Neural Models
### Robust04

|  Model   | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![status MatchZoo]][matchzoo code DSSM]  | [0.095][paper DRMM]  | [0.171][paper DRMM] | [0.201][paper DRMM]  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM  | [![status MatchZoo]][matchzoo code CDSSM] |[0.067][paper DRMM] | [0.125][paper DRMM] | [0.146][paper DRMM] |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I  | [![status MatchZoo]][matchzoo code ARC-I] |[0.041][paper DRMM] | [0.065][paper DRMM] | [0.066][paper DRMM] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II  | [![status MatchZoo]][matchzoo code ARC-II] |[0.067][paper DRMM] | [0.128][paper DRMM] | [0.147][paper DRMM] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM  | [![status MatchZoo]][matchzoo code DRMM]| 0.279| 0.431 | 0.382 | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid]| \\ | \\ | \\ | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM  | [![status MatchZoo]][matchzoo code KNRM] | \\ |[0.352][paper CEDR]|[0.409][paper CEDR] | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]  | \\ | \\ |[0.416][paper BertForIR]| [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet  | [![status MatchZoo]][matchzoo code Duet] | \\ | \\ | \\ |[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR  | ![status not in plan] | \\ | \\ | \\ | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank  | ![status not in plan] | \\ | \\ | \\ | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT  | ![status not in plan]| \\ | \\ | \\ |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |

### ClueWeb09-B
|  Model      | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----       | ----  |----  | ----  | ----  | ----  |
|  DSSM       | [![status MatchZoo]][matchzoo code DSSM]         | [0.054][paper DRMM]  | [0.185][paper DRMM] | [0.132][paper DRMM]  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM       | [![status MatchZoo]][matchzoo code CDSSM]        | [0.064][paper DRMM] | [0.214][paper DRMM] | [0.153][paper DRMM] |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I       | [![status MatchZoo]][matchzoo code ARC-I]        | [0.024][paper DRMM] | [0.089][paper DRMM] | [0.073][paper DRMM] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II      | [![status MatchZoo]][matchzoo code ARC-II]       | [0.033][paper DRMM] | [0.123][paper DRMM] | [0.087][paper DRMM] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM       | [![status MatchZoo]][matchzoo code DRMM]         | 0.133| 0.365 | 0.258 | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid] | \\ | \\ | \\ | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM       | [![status MatchZoo]][matchzoo code KNRM]         | \\ | \\ | \\ | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]    | \\ | \\ |[0.270][paper BertForIR]| [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet       | [![status MatchZoo]][matchzoo code Duet]         | \\ | \\ | \\ |[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR   | ![status not in plan]                            | \\ | \\ | \\ | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank   | ![status not in plan]                            | \\ | \\ | \\ | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT       | ![status not in plan]                            | \\ | \\ | \\ |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |

### Gov2
|  Model   | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![status MatchZoo]][matchzoo code DSSM]  | [todo](todo)  | [todo](todo) | [todo](todo)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM  | [![status MatchZoo]][matchzoo code CDSSM] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I  | [![status MatchZoo]][matchzoo code ARC-I] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II  | [![status MatchZoo]][matchzoo code ARC-II] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM  | [![status MatchZoo]][matchzoo code DRMM]| [todo](todo)  | [todo](todo) | [todo](todo)  | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid]| [todo](todo)  | [todo](todo) | [todo](todo)  | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM  | [![status MatchZoo]][matchzoo code KNRM] | [todo](todo)  | [todo](todo) | [todo](todo)  | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]  | [todo](todo)  | [todo](todo) | [todo](todo)  | [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet  | [![status MatchZoo]][matchzoo code Duet] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR  | ![status not in plan] | [todo](todo)  | [todo](todo) | [todo](todo)  | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank  | ![status not in plan] | [todo](todo)  | [todo](todo) | [todo](todo)  | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT  | ![status not in plan]| [todo](todo)  | [todo](todo) | [todo](todo)  |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |


### MS MARCO
|  Model   | Code  | MAP | P@20 | nDCG@20| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![status MatchZoo]][matchzoo code DSSM]  | [todo](todo)  | [todo](todo) | [todo](todo)  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM  | [![status MatchZoo]][matchzoo code CDSSM] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I  | [![status MatchZoo]][matchzoo code ARC-I] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II  | [![status MatchZoo]][matchzoo code ARC-II] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM  | [![status MatchZoo]][matchzoo code DRMM]| [todo](todo)  | [todo](todo) | [todo](todo)  | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid]| [todo](todo)  | [todo](todo) | [todo](todo)  | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM  | [![status MatchZoo]][matchzoo code KNRM] | [todo](todo)  | [todo](todo) | [todo](todo)  | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]  | [todo](todo)  | [todo](todo) | [todo](todo)  | [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet  | [![status MatchZoo]][matchzoo code Duet] | [todo](todo)  | [todo](todo) | [todo](todo)  |[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR  | ![status not in plan] | [todo](todo)  | [todo](todo) | [todo](todo)  | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank  | ![status not in plan] | [todo](todo)  | [todo](todo) | [todo](todo)  | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT  | ![status not in plan]| [todo](todo)  | [todo](todo) | [todo](todo)  |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |


### MQ2007
|  Model   | Code  | MAP | P@10 | nDCG@10| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![status MatchZoo]][matchzoo code DSSM]  | [0.409][paper HiNT]  | [0.352][paper HiNT] | [0.371][paper HiNT]  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM  | [![status MatchZoo]][matchzoo code CDSSM] |[0.364][paper DeepRank]| [0.291][paper DeepRank] |[0.325][paper DeepRank]|[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I  | [![status MatchZoo]][matchzoo code ARC-I] |[0.417][paper DeepRank] | [0.364][paper DeepRank] | [0.386][paper DeepRank] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II  | [![status MatchZoo]][matchzoo code ARC-II] |[0.421][paper DeepRank] | [0.366][paper DeepRank] | [0.390][paper DeepRank] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM  | [![status MatchZoo]][matchzoo code DRMM]| [0.467][paper DeepRank]| [0.388][paper HiNT] | [0.440][paper HiNT] | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid]| [0.434][paper DeepRank]|[0.371][paper DeepRank] |[0.409][paper DeepRank] | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM  | [![status MatchZoo]][matchzoo code KNRM] | \\ | \\ | \\ | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]  | \\ | \\ | \\ | [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet  | [![status MatchZoo]][matchzoo code Duet] |[0.474][paper HiNT] |[0.398][paper HiNT]|[0.453][paper HiNT]|[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR  | ![status not in plan] | \\ | \\ | \\ | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank  | ![status not in plan] |0.497 |0.412|0.482 | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT  | ![status not in plan]|0.502 |0.447|0.490 |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |

### MQ2008
|  Model   | Code  | MAP | P@10 | nDCG@10| Paper | 
|  ----  | ----  |----  | ----  | ----  | ----  |
|  DSSM  | [![status MatchZoo]][matchzoo code DSSM]  | [0.391][paper HiNT]  | [0.221][paper HiNT] | [0.178][paper HiNT]  | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013] |
| CDSSM  | [![status MatchZoo]][matchzoo code CDSSM] |[0.395][paper DeepRank]| [0.222][paper DeepRank] |[0.175][paper DeepRank]|[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014] |
| ARC-I  | [![status MatchZoo]][matchzoo code ARC-I] |[0.424][paper DeepRank] | [0.311][paper DeepRank] | [0.187][paper DeepRank] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
| ARC-II  | [![status MatchZoo]][matchzoo code ARC-II] |[0.421][paper DeepRank] | [0.229][paper DeepRank] | [0.181][paper DeepRank] |[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014] |
|  DRMM  | [![status MatchZoo]][matchzoo code DRMM]| [0.473][paper DeepRank]| [0.245][paper HiNT] | [0.220][paper HiNT] | [A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016] |
| MatchPyramid| [![status MatchZoo]][matchzoo code MatchPyramid]| [0.449][paper DeepRank]|[0.239][paper DeepRank] |[0.211][paper DeepRank] | [Text Matching as Image Recognition, AAAI 2016] |
|  KNRM  | [![status MatchZoo]][matchzoo code KNRM] | \\ | \\ | \\ | [End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017] |
|  CONV-KNRM  | [![status MatchZoo]][matchzoo code CONV-KNRM]  | \\ | \\ | \\ | [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018] |
|  Duet  | [![status MatchZoo]][matchzoo code Duet] |[0.476][paper HiNT] |[0.240][paper HiNT]|[0.216][paper HiNT]|[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017] |
|  Co-PACRR  | ![status not in plan] | \\ | \\ | \\ | [Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018] |
|  DeepRank  | ![status not in plan] |0.498 |0.252|0.240 | [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017] |
|  HiNT  | ![status not in plan]|0.505 |0.255|0.244 |  [Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018] |



[status MatchZoo]: https://img.shields.io/badge/matchzoo-ready-green
[status official]: https://img.shields.io/badge/official-code-brightgreen
[status not in plan]: https://img.shields.io/badge/matchzoo-not%20in%20plan-red

[matchzoo code DSSM]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py  
[matchzoo code CDSSM]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/cdssm.py 
[matchzoo code ARC-I]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arci.py 
[matchzoo code ARC-II]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/arcii.py 
[matchzoo code DRMM]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm.py
[matchzoo code MatchPyramid]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/match_pyramid.py
[matchzoo code KNRM]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/knrm.py 
[matchzoo code CONV-KNRM]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py  
[matchzoo code Duet]: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/duet.py 
[matchzoo code Co-PACRR]: https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master 
[matchzoo code DeepRank]: https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master 
[matchzoo code HiNT]: https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match/blob/master

[official code DSSM]: todo  
[official code CDSSM]: todo 
[official code ARC-I]: todo 
[official code ARC-II]: todo 
[official code DRMM]: todo
[official code MatchPyramid]: todo
[official code KNRM]: todo 
[official code CONV-KNRM]: todo  
[official code Duet]: todo 
[official code Co-PACRR]: todo 
[official code DeepRank]: todo 
[official code HiNT]: todo 

[paper DSSM]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
[paper CDSSM]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf
[paper ARC-I]: https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf
[paper ARC-II]: https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf
[paper DRMM]: http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf
[paper MatchPyramid]: https://arxiv.org/pdf/1602.06359.pdf
[paper KNRM]: https://arxiv.org/pdf/1706.06613.pdf
[paper CONV-KNRM]: http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf
[paper Duet]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
[paper Co-PACRR]: https://arxiv.org/pdf/1706.10192.pdf
[paper DeepRank]: https://arxiv.org/pdf/1710.05649.pdf
[paper HiNT]: https://arxiv.org/pdf/1805.05737.pdf
[paper CEDR]: https://arxiv.org/pdf/1904.07094.pdf
[paper BertForIR]: https://arxiv.org/pdf/1905.09217.pdf

[Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, CIKM 2013]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf 
[Learning Semantic Representations Using Convolutional Neural Networks for Web Search, WWW 2014]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf 
[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014]: https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf 
[Convolutional Neural Network Architectures for Matching Natural Language Sentences, NIPS 2014]: https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf 
[A Deep Relevance Matching Model for Ad-hoc Retrieval, CIKM 2016]: http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf 
[Text Matching as Image Recognition, AAAI 2016]: https://arxiv.org/pdf/1602.06359.pdf 
[End-to-End Neural Ad-hoc Ranking with Kernel Pooling, SIGIR 2017]: https://arxiv.org/pdf/1706.06613.pdf 
[Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search, WSDM 2018]: http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf 
[Learning to Match using Local and Distributed Representations of Text for Web Search, WWW 2017]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf 
[Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval, WSDM 2018]: https://arxiv.org/pdf/1706.10192.pdf 
[DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval, CIKM 2017]: https://arxiv.org/pdf/1710.05649.pdf 
[Modeling Diverse Relevance Patterns in Ad-hoc Retrieval, SIGIR 2018]: https://arxiv.org/pdf/1805.05737.pdf 
[CEDR: Contextualized Embeddings for Document Ranking, SIGIR 2019]: https://arxiv.org/pdf/1904.07094.pdf
[Deeper Text Understanding for IR with Contextual Neural Language Modeling, SIGIR 2019]: https://arxiv.org/pdf/1905.09217.pdf