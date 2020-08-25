# Paraphrase Identification

---

**Paraphrase Identification** is an task to determine whether two sentences have the same meaning, a problem considered a touchstone of natural language understanding.

Take an instance in MRPC dataset for example, this is a pair of sentences with the same meaning:

sentence1: Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.

sentence2: Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.


Some benchmark datasets are listed in the following.

## Classic Datasets

|  Dataset    | pairs of sentence |
|  ----  | ----  |
|  [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2Fdefault.aspx)  | 5800  |
|  [SentEval](https://arxiv.org/pdf/1803.05449.pdf)/STS  | 1750 |
| [SentEval](https://arxiv.org/pdf/1803.05449.pdf)/SICK-R | 9840 | 
| [SentEval](https://arxiv.org/pdf/1803.05449.pdf)/SICK-E | 9840 | 
| [Quora Question Pair](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)| 404290 |]

- [**MRPC**](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2Fdefault.aspx) is short for Microsoft Research Paraphrase Corpus. It contains 5,800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship.
- [**SentEval**](https://arxiv.org/pdf/1803.05449.pdf) encompasses semantic relatedness datasets including
SICK and the STS Benchmark dataset. SICK dataset includes two subtasks SICK-R and SICK-E.
For STS and SICK-R, it learns to predict relatedness scores between a pair of sentences. For SICK-E, it has the same pairs of sentences with SICK-R but can be treated as a three-class classification problem (classes are 'entailment', 'contradiction’, and 'neutral').

- [**Quora Question Pairs**](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) is a task released by Quora which aims to identify duplicate questions. It consists of over 400,000 pairs of questions on Quora, and each question pair is annotated with a binary value indicating whether the two questions are paraphrase of each other.

A list of neural matching models for paraphrase identification models are as follows.



## MRPC

| Model | Code | accuracy | f1 | Paper|
| :-----:| :----: | :----: |:----: |:----: |
| XLNet-Large (ensemble) (Yang et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zihangdai/xlnet/) | 93.0  | 90.7 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
|  MT-DNN-ensemble (Liu et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn/) |92.7  | 90.3 |  [Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding](https://arxiv.org/pdf/1904.09482.pdf) | 
|  Snorkel MeTaL(ensemble) (Ratner et al., 2018) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/HazyResearch/metal) | 91.5  | 88.5 |  [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/pdf/1810.02840.pdf) | 
| GenSen (Subramanian et al., 2018)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Maluuba/gensen) | 78.6  | 84.4 |  [Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning](https://arxiv.org/abs/1804.00079) |
| InferSent (Conneau et al., 2017) |[![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/facebookresearch/InferSent)  | 76.2  | 83.1 |  [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) |
| TF-KLD (Ji and Eisenstein, 2013)  |  \ | 80.4 | 85.9 |  [Discriminative Improvements to Distributional Sentence Similarity](http://www.aclweb.org/anthology/D/D13/D13-1090.pdf) | 
| SpanBert (Joshi et al., 2019)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/facebookresearch/SpanBERT) | 90.9 | 87.9 | [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf) | 
|  MT-DNN (Liu et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn) | 91.1  | 88.2 | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf) | 
| AugDeepParaphrase (Agarwal et al., 2017)  | \  | 77.7 |  84.5 | [A Deep Network Model for Paraphrase Detection in Short Text Messages](https://arxiv.org/pdf/1712.02820.pdf) | 
| ERNIE (Zhang et al. 2019)  | [![official](https://img.shields.io/badge/official-code-brightgreen)]( https://github.com/thunlp/ERNIE) | 88.2  | \  | [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf) | 
| This work (Lan and Xu, 2018)  |  \  |  84.0  | \  | [Character-based Neural Networks for Sentence Pair Modeling](https://www.aclweb.org/anthology/N18-2025.pdf) |  
|  ABCNN (Yin et al.2018) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/yinwenpeng/Answer_Selection)  | 78.9 | 84.8 |  [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf) | 
| Attentive Tree-LSTMs (Zhou et al.2016)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/yoosan/sentpair)  | 75.8 |83.7 | [Modelling Sentence Pairs with Tree-structured Attentive Encoder](https://arxiv.org/pdf/1610.02806.pdf) | 
|  Bi-CNN-MI (Yin and Schutze, 2015) | \  | 78.1 | 84.4  | [Convolutional Neural Network for Paraphrase Identification](https://www.aclweb.org/anthology/N15-1091.pdf) |


## SentEval
The evaluation metric for STS and SICK-R is Pearson correlation.

The evaluation metri for SICK-E is classification accuracy.

| Model | Code |  SICK-R | SICK-E | STS | Paper|
| :-----:| :----: | :----: |:----: |:----: |:----: |
| XLNet-Large (ensemble) (Yang et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zihangdai/xlnet/) |  \ | \ |  	91.6  | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
|  MT-DNN-ensemble (Liu et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn/) | \  | \ |  91.1 |  [Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding](https://arxiv.org/pdf/1904.09482.pdf) | 
|  Snorkel MeTaL(ensemble) (Ratner et al., 2018) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/HazyResearch/metal) | \  |  \ |  90.1 |  [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/pdf/1810.02840.pdf) | 
| GenSen (Subramanian et al., 2018)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Maluuba/gensen) |  88.8 | 87.8  |  78.9	 |  [Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning](https://arxiv.org/abs/1804.00079) |
| InferSent (Conneau et al., 2017) |[![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/facebookresearch/InferSent)  | 88.4	  | 86.3 |  75.8 |  [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) |
| SpanBert (Joshi et al., 2019)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/facebookresearch/SpanBERT) | \  | \  |  89.9  | [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf) | 
|  MT-DNN (Liu et al., 2019) |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn) |  \  |  \  | 89.5  | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf) |  
| ERNIE (Zhang et al. 2019)  | [![official](https://img.shields.io/badge/official-code-brightgreen)]( https://github.com/thunlp/ERNIE) |  \  |  \  | 83.2 | [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf) | 
| PWIM (He and Lin, 2016) |[![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/lukecq1231/nli) |   \ | \  | 76.7 | [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://www.aclweb.org/anthology/N16-1108.pdf)|


## Quora Question Pair

| Model | Code | F1 | Accuracy | Paper|
| :-----:| :----: | :----: |:----: |:----: |
| XLNet-Large (ensemble) (Yang et al., 2019) |   [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zihangdai/xlnet/) | 74.2	 | 90.3 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
|  MT-DNN-ensemble (Liu et al., 2019) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn/) | 73.7 | 89.9 |  [Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding](https://arxiv.org/pdf/1904.09482.pdf) | 
|Snorkel MeTaL(ensemble) (Ratner et al., 2018)   |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/HazyResearch/metal)  | 73.1	  |  89.9  | [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/pdf/1810.02840.pdf) |
|MwAN (Tan et al., 2018) |  \ | \ | 89.12| [Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/proceedings/2018/0613.pdf) |
| DIIN (Gong et al., 2018)	  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/YichenGong/Densely-Interactive-Inference-Network) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/diin.py) | \  | 89.06 | [Natural Language Inference over Interaction Space](https://arxiv.org/pdf/1709.04348.pdf) | 
| pt-DecAtt (Char) (Tomar et al., 2017) | \ |  \  | 88.40  | [Neural Paraphrase Identification of Questions with Noisy Pretraining](https://arxiv.org/abs/1704.04565) |
| BiMPM (Wang et al., 2017)	  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/zhiguowang/BiMPM) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/bimpm.py) | \ | 88.17 | [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814) | 
| GenSen (Subramanian et al., 2018) | \  | \   | 87.01	  | [Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning](https://arxiv.org/abs/1804.00079)|  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Maluuba/gensen) |
|  This work (Wang et al.2016) | \ |  78.4 | 84.7  | [Sentence Similarity Learning by Lexical Decomposition and Composition](https://www.aclweb.org/anthology/C16-1127/) |  
| RE2 (Yang et al., 2019)  |  [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/hitvoice/RE2) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/re2.py)  | \ |  89.2 | [Simple and Effective Text Matching with Richer Alignment Features](https://www.aclweb.org/anthology/P19-1465.pdf) |
|  MSEM (Wang et al.2016) | \ | \  | 88.86  | [Multi-task Sentence Encoding Model for Semantic Retrieval in Question Answering Systems](https://arxiv.org/ftp/arxiv/papers/1911/1911.07405.pdf) | 
| Bi-CAS-LSTM (Choi et al.2019) | \ | \ |   88.6  | [Cell-aware Stacked LSTMs for Modeling Sentences](https://arxiv.org/pdf/1809.02279.pdf)|
|DecAtt (Parikh et al., 2016)| \ | \  | 86.5 | [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/pdf/1606.01933.pdf)| \ |
