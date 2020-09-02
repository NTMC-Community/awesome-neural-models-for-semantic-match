# Community Question Answering

**Community Question Answer** is to automatically search for relevant answers among many responses provided for a given question (Answer Selection), and search for relevant questions to reuse their existing answers (Question Retrieval).

## Classic Datasets

<table style="width: 500px; margin-left: auto; margin-right: auto;">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Domain</th>
      <th>#Question</th>
      <th>#Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://trec.nist.gov/data/qa.html"><strong>TRECQA</strong></a></td>
      <td>Open-domain</td>
      <td>1,229</td>
      <td>5,3417</td>
    </tr>
    <tr>
      <td><a href="https://www.microsoft.com/en-us/download/details.aspx?id=52419"><strong>WikiQA</strong></a></td>
      <td>Open-domain</td>
      <td>3,047</td>
      <td>29,258</td>
    </tr>
    <tr>
      <td><a href="https://github.com/shuzi/insuranceQA"><strong>InsuranceQA</strong></a></td>
      <td>Insurance</td>
      <td>12,889</td>
      <td>21,325</td>
    </tr>
    <tr>
      <td><a href="https://sites.google.com/view/fiqa"><strong>FiQA</strong></a></td>
      <td>Financial</td>
      <td>6,648</td>
      <td>57,641</td>
    </tr>
    <tr>
      <td><a href="https://webscope.sandbox.yahoo.com"><strong>Yahoo! Answers</strong></a></td>
      <td>Open-domain</td>
      <td>50,112</td>
      <td>253,440</td>
    </tr>
    <tr>
      <td><a href="http://alt.qcri.org/semeval2015/task3/"><strong>SemEval-2015 Task 3</strong></a></td>
      <td>Open-domain</td>
      <td>2,600</td>
      <td>16,541</td>
    </tr>
    <tr>
      <td><a href="http://alt.qcri.org/semeval2016/task3/"><strong>SemEval-2016 Task 3</strong></a></td>
      <td>Open-domain</td>
      <td>4,879</td>
      <td>36,198</td>
    </tr>
    <tr>
      <td><a href="http://alt.qcri.org/semeval2017/task3/"><strong>SemEval-2017 Task 3</strong></a></td>
      <td>Open-domain</td>
      <td>4,879</td>
      <td>36,198</td>
    </tr>
  </tbody>
</table>

- [**TRECQA**](https://trec.nist.gov/data/qa.html) dataset is created by [Wang et. al.](https://www.aclweb.org/anthology/D07-1003) from TREC QA track 8-13 data, with candidate answers automatically selected from each question’s document pool using a combination of overlapping non-stop word counts and pattern matching. This data set is one of the most widely used benchmarks for [answer sentence selection](<https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)>).

- [**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering by Microsoft Research.

- [**InsuranceQA**](https://github.com/shuzi/insuranceQA) is a non-factoid QA dataset from the insurance domain. Question may have multiple correct answers and normally the questions are much shorter than the answers. The average length of questions and answers in tokens are 7 and 95, respectively. For each question in the development and test sets, there is a set of 500 candidate answers.

- [**FiQA**](https://sites.google.com/view/fiqa) is a non-factoid QA dataset from the financial domain which has been recently released for WWW 2018 Challenges. The dataset is built by crawling Stackexchange, Reddit and StockTwits in which part of the questions are opinionated, targeting mined opinions and their respective entities, aspects, sentiment polarity and opinion holder.

- [**Yahoo! Answers**](https://webscope.sandbox.yahoo.com) is a web site where people post questions and answers, all of which are public to any web user willing to browse or download them. The data we have collected is the Yahoo! Answers corpus as of 10/25/2007. This is a benchmark dataset for communitybased question answering that was collected from Yahoo Answers. In this dataset, the answer lengths are relatively longer than TrecQA and WikiQA.

- [**SemEval-2015 Task 3**](http://alt.qcri.org/semeval2015/task3/) consists of two sub-tasks. In Subtask A, given a question (short title + extended description), and several community answers, classify each of the answer as definitely relevance (good), potentially useful (potential), or bad or irrelevant (bad, dialog, non-english other). In Subtask B, given a YES/NO question (short title + extended description), and a list of community answers, decide whether the global answer to the question should be yes, no, or unsure.

- [**SemEval-2016 Task 3**](http://alt.qcri.org/semeval2016/task3/) consists two sub-tasks, namely _Question-Comment Similarity_ and _Question-Question Similarity_. In the _Question-Comment Similarity_ task, given a question from a question-comment thread, rank the comments according to their relevance with respect to the question. In _Question-Question Similarity_ task, given the new question, rerank all similar questions retrieved by a search engine.

- [**SemEval-2017 Task 3**](http://alt.qcri.org/semeval2017/task3/) contains two sub-tasks, namely _Question Similarity_ and _Relevance Classification_. Given the new question and a set of related questions from the collection, the _Question Similarity_ task is to rank the similar questions according to their similarity to the original question. While the _Relevance Classification_ is to rank the answer posts according to their relevance with respect to the question based on a question-answer thread.

## Performance

### TREC QA (Raw Version)

| Model                               | Code                                                         |    MAP    |    MRR    | Paper                                                        |
| :---------------------------------- | :----------------------------------------------------------: | :-------: | :-------: | :----------------------------------------------------------- |
| Punyakanok (2004)                   | —                                                          |   0.419   |   0.494   | [Mapping dependencies trees: An application to question answering, ISAIM 2004](http://cogcomp.cs.illinois.edu/papers/PunyakanokRoYi04a.pdf) |
| Cui (2005)                          | —                                                          |   0.427   |   0.526   | [Question Answering Passage Retrieval Using Dependency Relations, SIGIR 2005](https://www.comp.nus.edu.sg/~kanmy/papers/f66-cui.pdf) |
| Wang (2007)                         | —                                                          |   0.603   |   0.685   | [What is the Jeopardy Model? A Quasi-Synchronous Grammar for QA, EMNLP 2007](http://www.aclweb.org/anthology/D/D07/D07-1003.pdf) |
| H&S (2010)                          | —                                                          |   0.609   |   0.692   | [Tree Edit Models for Recognizing Textual Entailments, Paraphrases, and Answers to Questions, NAACL 2010](http://www.aclweb.org/anthology/N10-1145) |
| W&M (2010)                          | —                                                          |   0.595   |   0.695   | [Probabilistic Tree-Edit Models with Structured Latent Variables for Textual Entailment and Question Answering, COLING 2020](http://aclweb.org/anthology//C/C10/C10-1131.pdf) |
| Yao (2013)                          | —                                                          |   0.631   |   0.748   | [Answer Extraction as Sequence Tagging with Tree Edit Distance, NAACL 2013](http://www.aclweb.org/anthology/N13-1106.pdf) |
| S&M (2013)                          | —                                                          |   0.678   |   0.736   | [Automatic Feature Engineering for Answer Selection and Extraction, EMNLP 2013](http://www.aclweb.org/anthology/D13-1044.pdf) |
| Backward (Shnarch et al., 2013)     | —                                                          |   0.686   |   0.754   | [Probabilistic Models for Lexical Inference, Ph.D. thesis 2013](http://u.cs.biu.ac.il/~nlp/wp-content/uploads/eyal-thesis-library-ready.pdf) |
| LCLR (Yih et al., 2013)             | —                                                          |   0.709   |   0.770   | [Question Answering Using Enhanced Lexical Semantic Models, ACL 2013](http://research.microsoft.com/pubs/192357/QA-SentSel-Updated-PostACL.pdf) |
| bigram+count (Yu et al., 2014)      | —                                                          |   0.711   |   0.785   | [Deep Learning for Answer Sentence Selection, NIPS 2014](http://arxiv.org/pdf/1412.1632v1.pdf) |
| BLSTM (W&N et al., 2015)            | —                                                          |   0.713   |   0.791   | [A Long Short-Term Memory Model for Answer Sentence Selection in Question Answering, ACL 2015](http://www.aclweb.org/anthology/P15-2116) |
| Architecture-II (Feng et al., 2015) | —                                                          |   0.711   |   0.800   | [Applying deep learning to answer selection: A study and an open task, ASRU 2015](http://arxiv.org/abs/1508.01585) |
| PairCNN (Severyn et al., 2015)      | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zhangzibin/PairCNN-Ranking) |   0.746   |   0.808   | [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks, SIGIR 2015](http://disi.unitn.eu/moschitti/since2013/2015_SIGIR_Severyn_LearningRankShort.pdf) |
| aNMM (Yang et al., 2016)            | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/yangliuy/aNMM-CIKM16) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/anmm.py) |   0.750   |   0.811   | [aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model, CIKM 2016](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240) |
| HDLA (Tay et al., 2017)             | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/vanzytay/YahooQA_Splits) |   0.750   |   0.815   | [Learning to Rank Question Answer Pairs with Holographic Dual LSTM Architecture, SIGIR 2017](https://arxiv.org/abs/1707.06372) |
| PWIM (Hua et al. 2016)              | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/VDPWI-NN-Torch) |   0.758   |   0.822   | [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement, NAACL 2016](https://cs.uwaterloo.ca/~jimmylin/publications/He_etal_NAACL-HTL2016.pdf) |
| MP-CNN (Hua et al. 2015)            | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/MP-CNN-Torch) |   0.762   |   0.830   | [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks, EMNLP 2015](http://aclweb.org/anthology/D/D15/D15-1181.pdf) |
| HyperQA (Tay et al., 2017)          | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/vanzytay/WSDM2018_HyperQA) |   0.770   |   0.825   | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks, WSDM 2018](https://arxiv.org/pdf/1707.07847) |
| MP-CNN (Rao et al., 2016)           | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/NCE-CNN-Torch) |   0.780   |   0.834   | [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks, CIKM 2016](https://dl.acm.org/authorize.cfm?key=N27026) |
| HCAN (Rao et al., 2019)             | —                                                          |   0.774   |   0.843   | [Bridging the Gap between Relevance Matching and Semantic Matching for Short Text Similarity Modeling, EMNLP 2019](https://jinfengr.github.io/publications/Rao_etal_EMNLP2019.pdf) |
| MP-CNN (Tayyar et al., 2018)        | —                                                          |   0.836   |   0.863   | [Integrating Question Classification and Deep Learning for improved Answer Selection, COLING 2018](https://aclanthology.coli.uni-saarland.de/papers/C18-1278/c18-1278) |
| Pre-Attention (Kamath et al., 2019) | —                                                          |   0.852   |   0.891   | [Predicting and Integrating Expected Answer Types into a Simple Recurrent Neural Network Model for Answer Sentence Selection, CICLING 2019](https://hal.archives-ouvertes.fr/hal-02104488/) |
| CETE (Laskar et al., 2020)          | —                                                          | **0.950** | **0.980** | [Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task LREC 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.676.pdf) |

### TREC QA (Clean Version)

| Model                            | Code                                                         |    MAP    |    MRR    | Paper                                                        |
| :------------------------------- | :----------------------------------------------------------: | :-------: | :-------: | :----------------------------------------------------------- |
| W&I (2015)                       | —                                                          |   0.746   |   0.820   | [FAQ-based Question Answering via Word Alignment, arXiv 2015](http://arxiv.org/abs/1507.02628) |
| LSTM (Tan et al., 2015)          | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Alan-Lee123/answer-selection) |   0.728   |   0.832   | [LSTM-Based Deep Learning Models for Nonfactoid Answer Selection, arXiv 2015](http://arxiv.org/abs/1511.04108) |
| AP-CNN (dos Santos et al. 2016)  | —                                                          |   0.753   |   0.851   | [Attentive Pooling Networks, arXiv 2016](http://arxiv.org/abs/1602.03609) |
| L.D.C Model (Wang et al., 2016)  | —                                                          |   0.771   |   0.845   | [Sentence Similarity Learning by Lexical Decomposition and Composition, COLING 2016](http://arxiv.org/pdf/1602.07019v1.pdf) |
| MP-CNN (Hua et al., 2015)        | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/MP-CNN-Torch) |   0.777   |   0.836   | [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks, EMNLP 2015](http://aclweb.org/anthology/D/D15/D15-1181.pdf) |
| HyperQA (Tay et al., 2017)       | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/vanzytay/WSDM2018_HyperQA) |   0.784   |   0.865   | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks, WSDM 2018](https://arxiv.org/pdf/1707.07847) |
| MP-CNN (Rao et al., 2016)        | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/NCE-CNN-Torch) |   0.801   |   0.877   | [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks, CIKM 2016](https://dl.acm.org/authorize.cfm?key=N27026) |
| BiMPM (Wang et al., 2017)        | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zhiguowang/BiMPM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/bimpm.py) |   0.802   |   0.875   | [Bilateral Multi-Perspective Matching for Natural Language Sentences, arXiv 2017](https://arxiv.org/pdf/1702.03814.pdf) |
| CA (Bian et al., 2017)           | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/wjbianjason/Dynamic-Clip-Attention) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/dynamic_clip.py) |   0.821   |   0.899   | [A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection, CIKM 2017](https://dl.acm.org/citation.cfm?id=3133089&CFID=791659397&CFTOKEN=43388059) |
| IWAN (Shen et al., 2017)         | —                                                          |   0.822   |   0.889   | [Inter-Weighted Alignment Network for Sentence Pair Modeling, EMNLP 2017](https://aclanthology.info/pdf/D/D17/D17-1122.pdf) |
| sCARNN (Tran et al., 2018)       | —                                                          |   0.829   |   0.875   | [The Context-dependent Additive Recurrent Neural Net, NAACL 2018](http://www.aclweb.org/anthology/N18-1115) |
| MCAN (Tay et al., 2018)          | —                                                          |   0.838   |   0.904   | [Multi-Cast Attention Networks, KDD 2018](https://arxiv.org/abs/1806.00778) |
| MP-CNN (Tayyar et al., 2018)     | —                                                          |   0.865   |   0.904   | [Integrating Question Classification and Deep Learning for improved Answer Selection, COLING 2018](https://aclanthology.coli.uni-saarland.de/papers/C18-1278/c18-1278) |
| CA + LM + LC (Yoon et al., 2019) | —                                                          |   0.868   |   0.928   | [A Compare-Aggregate Model with Latent Clustering for Answer Selection, CIKM 2019](https://arxiv.org/abs/1905.12897) |
| GSAMN (Lai et al., 2019)         | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/laituan245/StackExchangeQA) |   0.914   |   0.957   | [A Gated Self-attention Memory Network for Answer Selection, EMNLP 2019](https://arxiv.org/pdf/1909.09696.pdf) |
| TANDA (Garg et al., 2019)        | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/alexa/wqa_tanda) | **0.943** |   0.974   | [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection, AAAI 2020](https://arxiv.org/abs/1911.04118) |
| CETE (Laskar et al., 2020)       | —                                                          |   0.936   | **0.978** | [Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task, LREC 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.676.pdf) |

### WikiQA

| Model                                    | Code                                                                                                                                                                                                                                                                                  | MAP       | MRR       | Paper                                                                                                                                                               |
| ---------------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ABCNN (Yin et al., 2016)                 | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/galsang/ABCNN)                                                                                                                                                                               | 0.6921    | 0.7108    | [ABCNN: Attention-based convolutional neural network for modeling sentence pairs, ACL 2016](https://doi.org/10.1162/tacl_a_00097)                                   |
| Multi-Perspective CNN (Rao et al., 2016) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/castorini/NCE-CNN-Torch)                                                                                                                                                                     | 0.701     | 0.718     | [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks, CIKM 2016](https://dl.acm.org/authorize.cfm?key=N27026)                               |
| HyperQA (Tay et al., 2017)               | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/vanzytay/WSDM2018_HyperQA)                                                                                                                                                                   | 0.705     | 0.720     | [Enabling Efficient Question Answer Retrieval via Hyperbolic Neural Networks, WSDM 2018](https://arxiv.org/pdf/1707.07847)                                          |
| KVMN (Miller et al., 2016)               | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/siyuanzhao/key-value-memory-networks)                                                                                                                                                        | 0.7069    | 0.7265    | [Key-Value Memory Networks for Directly Reading Documents, ACL 2016](https://doi.org/10.18653/v1/D16-1147)                                                          |
| BiMPM (Wang et al., 2017)                | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zhiguowang/BiMPM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/bimpm.py)                          | 0.718     | 0.731     | [Bilateral Multi-Perspective Matching for Natural Language Sentences, IJCAI 2017](https://arxiv.org/pdf/1702.03814.pdf)                                             |
| IWAN (Shen et al., 2017)                 | —                                                                                                                                                                                                           | 0.733     | 0.750     | [Inter-Weighted Alignment Network for Sentence Pair Modeling, EMNLP 2017](https://aclanthology.info/pdf/D/D17/D17-1122.pdf)                                         |
| CA (Wang and Jiang, 2017)                | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/pcgreat/SeqMatchSeq)                                                                                                                                                                         | 0.7433    | 0.7545    | [A Compare-Aggregate Model for Matching Text Sequences, ICLR 2017](https://arxiv.org/abs/1611.01747)                                                                |
| HCRN (Tay et al., 2018c)                 | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/hcrn.py)                                                                                                                                      | 0.7430    | 0.7560    | [Hermitian co-attention networks for text matching in asymmetrical domains, IJCAI 2018](https://www.ijcai.org/proceedings/2018/615)                                 |
| Compare-Aggregate (Bian et al., 2017)    | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/wjbianjason/Dynamic-Clip-Attention) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/dynamic_clip.py) | 0.748     | 0.758     | [A Compare-Aggregate Model with Dynamic-Clip Attention for Answer Selection, CIKM 2017](https://dl.acm.org/citation.cfm?id=3133089&CFID=791659397&CFTOKEN=43388059) |
| RE2 (Yang et al., 2019)                  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/alibaba-edu/simple-effective-text-matching) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/re2.py)  | 0.7452    | 0.7618    | [Simple and Effective Text Matching with Richer Alignment Features, ACL 2019](https://www.aclweb.org/anthology/P19-1465.pdf)                                        |
| GSAMN (Lai et al., 2019)                 | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/laituan245/StackExchangeQA)                                                                                                                                                                  | **0.857** | **0.872** | [A Gated Self-attention Memory Network for Answer Selection, EMNLP 2019](https://arxiv.org/pdf/1909.09696.pdf)                                                      |
