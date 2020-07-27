# Community Question Answering

**Community Question Answer** is to automatically search for relevant answers among many responses provided for a given question (Answer Selection), and search for relevant questions to reuse their existing answers (Question Retrieval).

### TREC QA

[**TRECQA**](https://trec.nist.gov/data/qa.html) dataset is created by [Wang et. al.](https://www.aclweb.org/anthology/D07-1003) from TREC QA track 8-13 data, with candidate answers automatically selected from each question’s document pool using a combination of overlapping non-stop word counts and pattern matching. This data set is one of the most widely used benchmarks for [answer sentence selection](https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)).

**Raw Version of TREC QA:**

| Model                                                        |    MAP    |    MRR    | Paper                                                        |
| :----------------------------------------------------------- | :-------: | :-------: | :----------------------------------------------------------- |
| Punyakanok (2004)                                            |   0.419   |   0.494   | [Mapping dependencies trees: An application to question answering](http://cogcomp.cs.illinois.edu/papers/PunyakanokRoYi04a.pdf) |
| Cui (2005)                                                   |   0.427   |   0.526   | [Question answering passage retrieval using dependency relations](http://ws.csie.ncku.edu.tw/login/upload/2005/paper/Question answering Question answering passage retrieval using dependency relations.pdf) |
| Wang (2007)                                                  |   0.603   |   0.685   | [What is the Jeopardy Model? A Quasi-Synchronous Grammar for QA](http://www.aclweb.org/anthology/D/D07/D07-1003.pdf) |
| H&S (2010)                                                   |   0.609   |   0.692   | [Tree Edit Models for Recognizing Textual Entailments, Paraphrases, and Answers to Questions](http://www.aclweb.org/anthology/N10-1145) |
| W&M (2010)                                                   |   0.595   |   0.695   | [Probabilistic Tree-Edit Models with Structured Latent Variables for Textual Entailment and Question Answering](http://aclweb.org/anthology//C/C10/C10-1131.pdf) |
| Yao (2013)                                                   |   0.631   |   0.748   | [Answer Extraction as Sequence Tagging with Tree Edit Distance](http://www.aclweb.org/anthology/N13-1106.pdf) |
| S&M (2013)                                                   |   0.678   |   0.736   | [Automatic Feature Engineering for Answer Selection and Extraction](http://www.aclweb.org/anthology/D13-1044.pdf)[](http://www.aclweb.org/anthology/N13-1106.pdf) |
| Shnarch (2013) - Backward                                    |   0.686   |   0.754   | Probabilistic Models for Lexical Inference                   |
| Yih (2013) - LCLR                                            |   0.709   |   0.770   | [Question Answering Using Enhanced Lexical Semantic Models](http://research.microsoft.com/pubs/192357/QA-SentSel-Updated-PostACL.pdf) |
| Yu (2014) - TRAIN-ALL bigram+count                           |   0.711   |   0.785   | [Deep Learning for Answer Sentence Selection](http://arxiv.org/pdf/1412.1632v1.pdf) |
| W&N (2015) - Three-Layer BLSTM+BM25                          |   0.713   |   0.791   | [A Long Short-Term Memory Model for Answer Sentence Selection in Question Answering](http://www.aclweb.org/anthology/P15-2116) |
| Feng (2015) - Architecture-II                                |   0.711   |   0.800   | [Applying deep learning to answer selection: A study and an open task](http://arxiv.org/abs/1508.01585) |
| S&M (2015)                                                   |   0.746   |   0.808   | [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf) |
| Yang (2016) - Attention-Based Neural Matching Model          |   0.750   |   0.811   | [aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240) |
| Tay (2017) - Holographic Dual LSTM Architecture              |   0.750   |   0.815   | [Learning to Rank Question Answer Pairs with Holographic Dual LSTM Architecture](https://arxiv.org/abs/1707.06372) |
| H&L (2016) - Pairwise Word Interaction Modelling             |   0.758   |   0.822   | [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://cs.uwaterloo.ca/~jimmylin/publications/He_etal_NAACL-HTL2016.pdf) |
| H&L (2015) - Multi-Perspective CNN                           |   0.762   |   0.830   | [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf) |
| Tay (2017) - HyperQA (Hyperbolic Embeddings)                 |   0.770   |   0.825   | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks](https://arxiv.org/pdf/1707.07847) |
| Rao (2016) - PairwiseRank + Multi-Perspective CNN            |   0.780   |   0.834   | [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/authorize.cfm?key=N27026) |
| Rao (2019) - Hybrid Co-Attention Network (HCAN)              |   0.774   |   0.843   | [Bridging the Gap between Relevance Matching and Semantic Matching for Short Text Similarity Modeling](https://jinfengr.github.io/publications/Rao_etal_EMNLP2019.pdf) |
| Tayyar Madabushi (2018) - Question Classification + PairwiseRank + Multi-Perspective CNN |   0.836   |   0.863   | [Integrating Question Classification and Deep Learning for improved Answer Selection](https://aclanthology.coli.uni-saarland.de/papers/C18-1278/c18-1278) |
| Kamath (2019) - Question Classification + RNN + Pre-Attention |   0.852   |   0.891   | [Predicting and Integrating Expected Answer Types into a Simple Recurrent Neural Network Model for Answer Sentence Selection](https://hal.archives-ouvertes.fr/hal-02104488/) |
| Laskar et al. (2020) - CETE (RoBERTa-Large)                  | **0.950** | **0.980** | [Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.676.pdf) |

**Clean Version of TREC QA:**

| Model                                                        |    MAP    |    MRR    | Paper                                                        |
| :----------------------------------------------------------- | :-------: | :-------: | :----------------------------------------------------------- |
| W&I (2015)                                                   |   0.746   |   0.820   | [FAQ-based Question Answering via Word Alignment](http://arxiv.org/abs/1507.02628) |
| Tan (2015) - QA-LSTM/CNN+attention                           |   0.728   |   0.832   | [LSTM-Based Deep Learning Models for Nonfactoid Answer Selection](http://arxiv.org/abs/1511.04108) |
| dos Santos (2016) - Attentive Pooling CNN                    |   0.753   |   0.851   | [[Attentive Pooling Networks](http://arxiv.org/abs/1602.03609)](http://www.aclweb.org/anthology/P15-2116) |
| Wang et al. (2016) - L.D.C Model                             |   0.771   |   0.845   | [Sentence Similarity Learning by Lexical Decomposition and Composition](http://arxiv.org/pdf/1602.07019v1.pdf) |
| H&L (2015) - Multi-Perspective CNN                           |   0.777   |   0.836   | [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](http://aclweb.org/anthology/D/D15/D15-1181.pdf) |
| Tay (2017) - HyperQA (Hyperbolic Embeddings)                 |   0.784   |   0.865   | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks](https://arxiv.org/pdf/1707.07847) |
| Rao (2016) - PairwiseRank + Multi-Perspective CNN            |   0.801   |   0.877   | [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks](https://dl.acm.org/authorize.cfm?key=N27026) |
| Wang et al. (2017) - BiMPM                                   |   0.802   |   0.875   | [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf) |
| Bian et al. (2017) - Compare-Aggregate                       |   0.821   |   0.899   | [A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection](https://dl.acm.org/citation.cfm?id=3133089&CFID=791659397&CFTOKEN=43388059) |
| Shen et al. (2017) - IWAN                                    |   0.822   |   0.889   | [Inter-Weighted Alignment Network for Sentence Pair Modeling](https://aclanthology.info/pdf/D/D17/D17-1122.pdf) |
| Tran et al. (2018) - IWAN + sCARNN                           |   0.829   |   0.875   | [The Context-dependent Additive Recurrent Neural Net](http://www.aclweb.org/anthology/N18-1115) |
| Tay et al. (2018) - Multi-Cast Attention Networks (MCAN)     |   0.838   |   0.904   | [Multi-Cast Attention Networks](https://arxiv.org/abs/1806.00778) |
| Tayyar Madabushi (2018) - Question Classification + PairwiseRank + Multi-Perspective CNN |   0.865   |   0.904   | [Integrating Question Classification and Deep Learning for improved Answer Selection](https://aclanthology.coli.uni-saarland.de/papers/C18-1278/c18-1278) |
| Yoon et al. (2019) - Compare-Aggregate + LanguageModel + LatentClustering |   0.868   |   0.928   | [[A Compare-Aggregate Model with Latent Clustering for Answer Selection](https://arxiv.org/abs/1905.12897)](https://hal.archives-ouvertes.fr/hal-02104488/) |
| Lai et al. (2019) - BERT + GSAMN + Transfer Learning         |   0.914   |   0.957   | [A Gated Self-attention Memory Network for Answer Selection](https://arxiv.org/pdf/1909.09696.pdf) |
| Garg et al. (2019) - TANDA-RoBERTa (ASNQ, TREC-QA)           | **0.943** |   0.974   | [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection](https://arxiv.org/abs/1911.04118) |
| Laskar et al. (2020) - CETE (RoBERTa-Large)                  |   0.936   | **0.978** | [Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.676.pdf) |

### WikiQA

[**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering by Microsoft Research.

| Model                                               | MAP    | MRR    | Paper                                                        |
| --------------------------------------------------- | ------ | ------ | ------------------------------------------------------------ |
| ABCNN (Yin et al., 2016)                            | 0.6921 | 0.7108 | [ABCNN: Attention-based convolutional neural network for modeling sentence pairs](https://doi.org/10.1162/tacl_a_00097) |
| HyperQA (Tay et al., 2017)                          | 0.705  | 0.720  | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks](https://arxiv.org/pdf/1707.07847) |
| KVMN (Miller et al., 2016)                          | 0.7069 | 0.7265 | [Key-value memory networks for directly reading documents](https://doi.org/10.18653/v1/D16-1147) |
| BiMPM (Wang et al., 2017)                           | 0.718  | 0.731  | [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf) |
| IWAN (Shen et al., 2017)                            | 0.733  | 0.750  | [Inter-Weighted Alignment Network for Sentence Pair Modeling](https://aclanthology.info/pdf/D/D17/D17-1122.pdf) |
| CA(Wang and Jiang, 2017)                            | 0.7433 | 0.7545 | [A compare aggregate model for matching text sequences](https://arxiv.org/abs/1611.01747) |
| HCRN (Tay et al., 2018c)                            | 0.7430 | 0.7560 | [Hermitian co-attention networks for text matching in asymmetrical domains](https://www.ijcai.org/proceedings/2018/615) |
| Compare-Aggregate (Bian et al., 2017)               | 0.748  | 0.758  | [A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection](https://dl.acm.org/citation.cfm?id=3133089&CFID=791659397&CFTOKEN=43388059) |
| RE2 (Yang et al., 2019)                             | 0.7452 | 0.7618 | [Simple and Effective Text Matching with Richer Alignment Features](https://www.aclweb.org/anthology/P19-1465.pdf) |
| BERT + GSAMN + Transfer Learning (Lai et al., 2019) | 0.857  | 0.872  | [A Gated Self-attention Memory Network for Answer Selection](https://arxiv.org/pdf/1909.09696.pdf) |


### SemEval-2015 Task 3

[**SemEval-2015 Task 3**](http://alt.qcri.org/semeval2015/task3/) consists of two sub-tasks. In Subtask A, given a question (short title + extended description), and several community answers, classify each of the answer as definitely relevance (good), potentially useful (potential), or bad or irrelevant (bad, dialog, non-english other). In Subtask B, given a YES/NO question (short title + extended description), and a list of community answers, decide whether the global answer to the question should be yes, no, or unsure.

### SemEval-2016 Task 3

[**SemEval-2016 Task 3**](http://alt.qcri.org/semeval2016/task3/) consists two sub-tasks, namely *Question-Comment Similarity* and *Question-Question Similarity*. In the *Question-Comment Similarity* task, given a question from a question-comment thread, rank the comments according to their relevance with respect to the question. In *Question-Question Similarity* task, given the new question, rerank all similar questions retrieved by a search engine.

### SemEval-2017 Task 3

[**SemEval-2017 Task 3**](http://alt.qcri.org/semeval2017/task3/) contains two sub-tasks, namely *Question Similarity* and *Relevance Classification*. Given the new question and a set of related questions from the collection, the *Question Similarity* task is to rank the similar questions according to their similarity to the original question. While the *Relevance Classification* is to rank the answer posts according to their relevance with respect to the question based on a question-answer thread.

### Yahoo! Answers

Yahoo! Answers is a web site where people post questions and answers, all of which are public to any web user willing to browse or download them. The data we have collected is the Yahoo! Answers corpus as of 10/25/2007.

This is a benchmark dataset for communitybased question answering that was collected from Yahoo Answers. In this dataset, the answer lengths are relatively longer than TrecQA and WikiQA.

### FiQA

Non-factoid QA dataset from the financial domain which has been recently released for WWW 2018 Challenges. The dataset is built by crawling Stackexchange, Reddit and StockTwits in which part of the questions are opinionated, targeting mined opinions and their respective entities, aspects, sentiment polarity and opinion holder.

### InsuranceQA

Non-factoid QA dataset from the insurance domain. Question may have multiple correct answers and normally the questions are much shorter than the answers. The average length of questions and answers in tokens are 7 and 95, respectively. For each question in the development and test sets, there is a set of 500 candidate answers.
