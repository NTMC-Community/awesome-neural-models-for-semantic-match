# Natural language Inference

**Natural Language Inference** is the task of determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise".

## Classic Datasets

| Dataset                                                      | # sentence pair |
| ------------------------------------------- | ----- |
| [**SNLI**](https://arxiv.org/abs/1508.05326)                 | 570K            |
| [**MultiNLI**](https://arxiv.org/abs/1704.05426)             | 433K            |
| [**SciTail**](http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf) | 27K             |

- [**SNLI**](https://arxiv.org/abs/1508.05326) is the short of Stanford Natural Language Inference, which has 570k human annotated sentence pairs. Thre premise data is draw from the captions of the Flickr30k corpus, and the hypothesis data is manually composed.
- [**MultiNLI**](https://arxiv.org/abs/1704.05426) is short of Multi-Genre NLI, which has 433k sentence pairs, whose collection process and task detail are modeled closely to SNLI. The premise data is collected from maximally broad range of genre of American English such as non-fiction genres (SLATE, OUP, GOVERNMENT, VERBATIM, TRAVEL), spoken genres (TELEPHONE, FACE-TO-FACE), less formal written genres (FICTION, LETTERS) and a specialized one for 9/11.
- [**SciTail**](http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf) entailment dataset consists of 27k. In contrast to the SNLI and MultiNLI, it was not crowd-sourced but created from sentences that already exist “in the wild”. Hypotheses were created from science questions and the corresponding answer candidates, while relevant web sentences from a large corpus were used as premises.

## Performance

### SNLI

| Model                                                        | Code| Accuracy                       | Paper                                                        |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Match-LSTM (Wang et al. ,2016)                               | [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/matchlstm.py) | 86.1                                | [Learning Natural Language Inference with LSTM](https://www.aclweb.org/anthology/N16-1170.pdf) |
| Decomposable (Parikh et al., 2016)                           | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) |86.3/86.8(Intra-sentence attention) | [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/pdf/1606.01933.pdf) |
| BiMPM (Wang et al., 2017)                                    | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://zhiguowang.github.io/) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/bimpm.py)| 86.9                                | [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf) |
| Shortcut-Stacked BiLSTM (Nie et al., 2017)                   | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/easonnie/multiNLI_encoder) | 86.1                                | [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference](https://arxiv.org/pdf/1708.02312.pdf) |
| ESIM (Chen et al., 2017)                                     | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/lukecq1231/nli) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/esim.py) |88.0/88.6(Tree-LSTM)                | [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf) |
| DIIN (Gong et al., 2018)                                     | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/YichenGong/Densely-Interactive-Inference-Network) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/diin.py) | 88.0                                | [Natural Language Inference over Interaction Space](https://arxiv.org/pdf/1709.04348.pdf) |
| SAN (Liu et al., 2018)                                       | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.7                                | [Stochastic Answer Networks for Natural Language Inference](https://arxiv.org/pdf/1804.07888.pdf) |
| AF-DMN (Duan et al., 2018)                                   | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.6                                | [Attention-Fused Deep Matching Network for Natural Language Inference](https://www.ijcai.org/Proceedings/2018/0561.pdf) |
| MwAN (Tan et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.3                                | [Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/Proceedings/2018/0613.pdf) |
| HBMP (Talman et al., 2018)                                   | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Helsinki-NLP/HBMP) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/hbmp.py) | 86.6                                | [Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture](https://arxiv.org/pdf/1808.08762v1.pdf) |
| CAFE (Tay et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.5                                | [Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference](https://arxiv.org/pdf/1801.00102v2.pdf) |
| DSA (Yoon et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 86.8                                | [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf) |
| Enhancing Sentence Embedding with Generalized Pooling (Chen et al., 2018) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/lukecq1231/generalized-pooling) | 86.6                                | [Enhancing Sentence Embedding with Generalized Pooling](https://arxiv.org/pdf/1806.09828.pdf?source=post_page---------------------------) |
| ReSAN (Shen et al., 2018)                                    | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/taoshen58/DiSAN/tree/master/ReSAN) | 86.3                                | [Reinforced Self-Attention Network: a Hybrid of Hard and Soft Attention for Sequence Modeling](https://arxiv.org/pdf/1801.10296.pdf) |
| DMAN (Pan et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.8                                | [Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference](https://www.aclweb.org/anthology/P18-1091.pdf) |
| DRCN (Kim et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 90.1                                | [Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information](https://www.aaai.org/ojs/index.php/AAAI/article/download/4627/4505) |
| RE2 (Yang et al., 2019)                                      | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/hitvoice/RE2) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/re2.py) | 88.9                                | [Simple and Effective Text Matching with Richer Alignment Features](https://arxiv.org/pdf/1908.00300.pdf) |
| MT-DNN (Liu et al., 2019)                                    | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn) | 91.1(base)/91.6(large)              | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf) |

### MNLI

| Model                                                        |Code|  Matched  Accuracy | Mismatched Accuracy | Paper                                                   |
| ------------------------------------------------------------ | ----------------------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ESIM (Chen et al., 2017)                                     | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/lukecq1231/nli) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/esim.py) | 76.8                    | 75.8                     | [Recurrent Neural Network-Based Sentence Encoder with Gated Attention for Natural Language Inference](https://arxiv.org/pdf/1708.01353.pdf) |
| Shortcut-Stacked BiLSTM (Nie et al., 2017)                   | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/easonnie/multiNLI_encoder) | 74.6                    | 73.6                     | [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference](https://arxiv.org/pdf/1708.02312.pdf) |
| HBMP (Talman et al., 2018)                                   | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Helsinki-NLP/HBMP) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/hbmp.py) | 73.7                    | 73.0                     | [Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture](https://arxiv.org/pdf/1808.08762v1.pdf) |
| Generalized Pooling (Chen et al., 2018) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/lukecq1231/generalized-pooling) | 73.8                    | 74.0                     | [Enhancing Sentence Embedding with Generalized Pooling](https://arxiv.org/pdf/1806.09828.pdf?source=post_page---------------------------) |
| AF-DMN (Duan et al., 2018)                                   |![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red)|  76.9                    | 76.3                     | [Attention-Fused Deep Matching Network for Natural Language Inference](https://www.ijcai.org/Proceedings/2018/0561.pdf) |
| DIIN (Gong et al., 2018)                                     | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/YichenGong/Densely-Interactive-Inference-Network)  [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/diin.py) | 78.8                    | 77.8                     | [Natural Language Inference over Interaction Space](https://github.com/YichenGong/Densely-Interactive-Inference-Network) |
| SAN (Liu et al., 2018)                                      | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 79.3                    | 78.7                     | [Stochastic Answer Networks for Natural Language Inference](https://arxiv.org/pdf/1804.07888.pdf) |
| MwAN (Tan et al., 2018)                                   | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 78.5                    | 77.7                     | [Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/Proceedings/2018/0613.pdf) |
| CAFE (Tay et al., 2018)                                  | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 78.7                    | 77.9                     | [Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference](https://arxiv.org/pdf/1801.00102v2.pdf) |
| DRCN (Kim et al., 2018)                                  | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 79.1                    | 78.4                     | [Semantic Sentence Matching with Densely-connected Recurrent and Co-attentive Information](https://www.aaai.org/ojs/index.php/AAAI/article/download/4627/4505) |
| DMAN (Pan et al., 2018)                                   | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 78.9                    | 78.2                     | [Discourse Marker Augmented Network with Reinforcement Learning for Natural Language Inference](https://www.aclweb.org/anthology/P18-1091.pdf) |


### SciTail

| Model                  | Code    | Accuracy               | Paper                                                        |
| -------------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SAN (Liu et al., 2018)     | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 88.4                   | [Stochastic Answer Networks for Natural Language Inference](https://arxiv.org/pdf/1804.07888.pdf) |                                               |
| HCRN (Tay et al., 2018)    | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 80.0                   | [Hermitian Co-Attention Networks for Text Matching in Asymmetrical Domains](https://www.ijcai.org/Proceedings/2018/0615.pdf) |                                               |
| HBMP (Talman et al., 2018) | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/Helsinki-NLP/HBMP) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/master/matchzoo/models/hbmp.py) | 86.0                   | [Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture](https://arxiv.org/pdf/1808.08762v1.pdf) |  |
| CAFE (Tay et al., 2018)    | ![not in plan](https://img.shields.io/badge/matchzoo-not%20in%20plan-red) | 83.3                   | [Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference](https://arxiv.org/pdf/1801.00102v2.pdf) |                                               |
| RE2 (Yang et al., 2019)    | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/hitvoice/RE2) [![MatchZoo](https://img.shields.io/badge/matchzoo-ready-green)](https://github.com/NTMC-Community/MatchZoo-py/blob/dev/matchzoo/models/re2.py) | 86.0                   | [Simple and Effective Text Matching with Richer Alignment Features](https://arxiv.org/pdf/1908.00300.pdf) |  |
| MT-DNN (Liu et al., 2019)  | [![official](https://img.shields.io/badge/official-code-brightgreen)](https://github.com/namisan/mt-dnn) | 94.1(base)/95.0(large) | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf) |                 |
















