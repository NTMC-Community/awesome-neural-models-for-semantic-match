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

​							$$\text{Match}(s, t) = g(f(\psi(s), \phi(t)))$$

Where $s$ and $t$ are source text input and target text input, respectively. The $\psi$ and $\phi$ are representation function for input $s$ and $t$, respectively. The $f$ is the interaction function, and $g$ is the aggregation function. The representative matching tasks are as follows:

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

* [**Gov2**](https://trec.nist.gov/data/terabyte.html) is a large Web collection which consists of 25 million documents in total. The queries are accumulated over TREC Terabyte Tracks 2004, 2005, and 2006. There are 150 queries in total.

* [**MSMARCO Passage Reranking**](http://www.msmarco.org/dataset.aspx) is a dataset collection from bing logs.

Neural information retrieval (NeuIR) models include:

1. **DSSM** [[code]]() [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) ![status](artworks/ready.svg)

   Learning Deep Structured Semantic Models for Web Search using Clickthrough Data. *CIKM 2013*.
   [website](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) | [tutorial](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/dl-summer-school-2017.-Jianfeng-Gao.v2.pdf)

2. **CDSSM**  [[code]]() [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf) ![status](artworks/ready.svg)

   Learning Semantic Representations Using Convolutional Neural Networks for Web Search. *WWW 2014*.
   [website](https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)

   A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. *CIKM 2014*.
   [website](https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf)

3. **DRMM**  [[code]]() [[pdf]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) ![status](artworks/ready.svg)

   A Deep Relevance Matching Model for Ad-hoc Retrieval.
   [website](https://arxiv.org/abs/1711.08611) | [paper](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)

4. **KNRM**  [[code]]() [[pdf]](https://arxiv.org/pdf/1706.06613.pdf) ![status](artworks/ready.svg)

   End-to-End Neural Ad-hoc Ranking with Kernel Pooling.
   [website](https://arxiv.org/abs/1706.06613) | [paper](https://arxiv.org/pdf/1706.06613.pdf)

5. **CONV-KNRM**  [[code]]() [[pdf]](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search.
   [website](https://dl.acm.org/citation.cfm?id=3159659) | [paper](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)

6. #### **Duet** [[code]]() [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) ![status](artworks/ready.svg)

   Learning to Match using Local and Distributed Representations of Text for Web Search.
   [website](https://dl.acm.org/citation.cfm?id=3052579) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf)

7. **Co-PACRR**  [[code]]() [[pdf]](https://arxiv.org/pdf/1706.10192.pdf) ![status](artworks/not-in-plan.svg)

   Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval.
   [website](https://arxiv.org/abs/1706.10192) | [paper](https://arxiv.org/pdf/1706.10192.pdf)

8. **LSTM-RNN**  [[code not ready]]() [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf) ![status](artworks/not-in-plan.svg)

   Deep Sentence Embsedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
   [website](https://arxiv.org/abs/1502.06922) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf)

9. **DRMM_TKS** [[code]]() [[pdf]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) ![status](/Users/eshion/Documents/Project/NTMC-Community/awaresome-neural-models-for-semantic-match/artworks/ready.svg)

   A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM).
   [website](https://arxiv.org/abs/1711.08611) | [paper](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)

10. **DeepRank**  [[code]]() [[pdf]](https://arxiv.org/pdf/1710.05649.pdf) ![status](artworks/progress.svg)

    DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval
    [website](https://arxiv.org/abs/1710.05649) | [paper](https://arxiv.org/pdf/1710.05649.pdf)

11. **HiNT**  [[code]]() [[pdf]](https://arxiv.org/pdf/1805.05737.pdf) ![status](artworks/progress.svg)

    Modeling Diverse Relevance Patterns in Ad-hoc Retrieval
    [website](https://arxiv.org/abs/1805.05737) | [paper](https://arxiv.org/pdf/1805.05737.pdf)



### Paraphrase Identification

---

**Paraphrase Identification** is an task to ...

Some benchmark datasets are listed in the following,

- [**AAA**]()
- [**BBB**]() 

A list of neural matching models for paraphrase identification models are as follows,

1. **ARC-I** [[code]]() [[pdf]] (https://arxiv.org/pdf/1503.03244.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Network Architectures for Matching Natural Language Sentences. NIPS 2014.
   [website](https://arxiv.org/abs/1503.03244) | [paper](https://arxiv.org/pdf/1503.03244.pdf)

2. **ARC-II** [[code]]() [[pdf]] (https://arxiv.org/pdf/1503.03244.pdf) ![status](artworks/ready.svg)

   Convolutional Neural Network Architectures for Matching Natural Language Sentences. NIPS 2014.
   [website](https://arxiv.org/abs/1503.03244) | [paper](https://arxiv.org/pdf/1503.03244.pdf)

3. **MV-LSTM** [[code]]() [[pdf]] (https://arxiv.org/pdf/1511.08277.pdf) ![status](artworks/ready.svg)

   A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations. AAAI 2016.
   [website](https://arxiv.org/abs/1511.08277) | [paper](https://arxiv.org/pdf/1511.08277.pdf)

4. **MatchPyramid** [[code]]() [[pdf]](https://arxiv.org/pdf/1602.06359.pdf) ![status](artworks/ready.svg)

   Text Matching as Image Recognition. AAAI 2016.
   [website](https://arxiv.org/abs/1602.06359) | [paper](https://arxiv.org/pdf/1602.06359.pdf)

5. **Match-SRNN**  [[code]]() [[pdf]] (https://arxiv.org/pdf/1604.04378.pdf) ![status](artworks/progress.svg)

   Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN. IJCAI 2016.
   [website](https://arxiv.org/abs/1604.04378) | [paper](https://arxiv.org/pdf/1604.04378.pdf)

6. **MultiGranCNN** [[code]]() [[pdf]](https://aclanthology.info/pdf/P/P15/P15-1007.pdf) ![status](artworks/not-in-plan.svg)

   MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity.
   [website](https://arxiv.org/abs/1604.04378) | [paper](https://aclanthology.info/pdf/P/P15/P15-1007.pdf)

7. **ABCNN** [[code]]() [[pdf]](https://arxiv.org/pdf/1512.05193.pdf) ![status](artworks/not-in-plan.svg)

   ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs.
   [website](https://arxiv.org/abs/1512.05193) | [paper](https://arxiv.org/pdf/1512.05193)

8. **…**

   

### Community Question Answer

---

**Community Question Answer** is the task for ...

Some benchmark datasets are listed in the following,

- [**AAA**]()
- [**BBB**]()
- 

Representative neural matching models for CQA include:

1. **aNMM** [[code]]() [[pdf]]() ![status](artworks/ready.svg)

   aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.
   [website](https://arxiv.org/abs/1801.01641) | [paper](http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240)

2. **MCAN** [[code]]() [[pdf]]() ![status](artworks/not-in-plan.svg)

   Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction
   [website](https://arxiv.org/abs/1806.00778) | [paper](https://arxiv.org/pdf/1806.00778.pdf)

3. **MIX** [[code]]() [[pdf]](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/hchen-kdd18.pdf) ![status](artworks/not-in-plan.svg)

   MIX: Multi-Channel Information Crossing for Text Matching
   [website](http://www.kdd.org/kdd2018/accepted-papers/view/mix-multi-channel-information-crossing-for-text-matching) | [paper](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/hchen-kdd18.pdf)

4. **…**

   



### Natural Language Inference

---

**Natural Language Inference** is the task to ...

Some benchmark datasets are listed in the following,

- [**AAA**]()
- [**BBB**]() 

Representative neural matching models for NLI include:

1. **Match-LSTM** [[code]]() [[pdf]]() ![status](artworks/ready.svg)

   Learning Natural Language Inference with LSTM
   [website](https://arxiv.org/abs/1512.08849) | [paper](http://www.aclweb.org/anthology/N16-1170)

2. **BiMPM** [[code]]() [[pdf]]() ![status](artworks/ready.svg)

   Bilateral Multi-Perspective Matching for Natural Language Sentences
   [website](https://arxiv.org/abs/1702.03814) | [paper](https://arxiv.org/pdf/1702.03814.pdf)

3. **ESIM** [[code]]() [[pdf]]() ![status](artworks/progress.svg)

   Enhanced LSTM for Natural Language Inference. ACL 2017
   [website](https://arxiv.org/abs/1609.06038) | [paper](https://arxiv.org/pdf/1609.06038.pdf)




### Healthcheck

```python
pip3 install -r requirements.txt
python3 healthcheck.py
```