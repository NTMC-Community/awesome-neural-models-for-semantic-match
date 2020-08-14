## Response retrieval
Response retrieval/selection aims to rank/select a proper response from a dialog repository.
Automatic conversation (AC) aims to create an automatic human-computer dialog process for the purpose of question answering, task completion, and social chat (i.e., chit-chat). In general, AC could be formulated either as an IR problem that aims to rank/select a proper response from a dialog repository or a generation problem that aims to generate an appropriate response with respect to the input utterance. Here, we refer response retrieval as the IR-based way to do AC.
Example:
![](https://codimd.s3.shivering-isles.com/demo/uploads/upload_8b357e5358f6a9398092d46ccfeb619a.png)



### Dataset

|Dataset |partition | #context response pair| #candidate per context | positive:negative |Avg #turns per context|
|---- | ---- | ---- |---- | ---- | ---- |
|UDC| train| 1M| 2| 1:1| 10.13|
|UDC| validation | 500k | 10| 1:9|10.11|
|UDC| test| 500k | 10| 1:9 | 10.11|
|Douban| train| 1M |2|1:1|6.69|
|Douban|validation| 50k| 2| 1:1| 6.75|
|Douban| test| 10k| 10| 1.18:8.82| 6.45|
|MSDialog| train| 173k | 10|1:9|5.0|
|MSDialog| validation| 37k| 10| 1:9| 4.9|
|MSDialog| test | 35k| 10 |1:9| 4.4
|EDC| train| 1M| 2| 1:1| 5.51|
|EDC| validation| 10k| 2| 1:1| 5.48|
|EDC| test | 10k| 10 | 1:9 | 5.64|

- Ubuntu Dialog Corpus (UDC) contains multi-turn dialogues collected from chat logs of the Ubuntu Forum. The data set consists of 1 million context-response pairs for training, 0.5 million pairs for validation, and 0.5 million pairs for testing. Positive responses are true responses from humans, and negative ones are randomly sampled. The ratio of the positive and the negative is 1:1 in training, and 1:9 in validation and testing. 
- [Douban Conversation Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) is an open domain dataset constructed from Douban group (a popular social networking service in China). The data set consists of 1 million context-response pairs for training, 50k pairs for validation, and 10k pairs for testing, corresponding to 2, 2, and 10 response candidates per context respectively. Response candidates on the test set, retrieved from Sina Weibo (the largest microblogging service in China), are labeled by human judges.
- [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) is a labeled dialog dataset of question answering (QA) interactions between information seekers and answer providers from an online forum on Microsoft products (Microsoft Community). The dataset contains more than 2,000 multi-turn information-seeking conversations with 10,000 utterances that are annotated with user intent on the utterance level.
- [E-commerce Dialogue Corpus](https://github.com/cooelf/DeepUtteranceAggregation)  contains over 5 types of conversations (e.g. commodity consultation, logistics express, recommendation, negotiation and chitchat) based on over 20 commodities. The ratio of the positive and the negative
is 1:1 in training and validation, and 1:9 in testing.

$R_n@k$: recall at position $k$ in $n$ candidates.

Ubuntu Corpus
| Model   | Code|MAP|$R_2@1$|$R_{10}@1$|$R_{10}@2$|$R_{10}@5$|Paper| type |
|  ----  | ----  |  ----  | --- | --- | --- | --- | ----  | ----  |
| Multi-View (Zhou et al. 2016)|  | \ | 0.908 | 0.662 | 0.801 | 0.951 |  [Multi-view Response Selection for Human-Computer Conversation](https://www.aclweb.org/anthology/D16-1036.pdf) | multi-turn |
| DL2R (Yan, Song and Wu 2016)| | \ |	0.899 |	0.626 |	0.783 |	0.944|[Learning to Respond with Deep Neural Networks for Retrieval-Based Human-Computer Conversation System](http://www.ruiyan.me/pubs/SIGIR2016.pdf) | multi-turn|
| SMN (Wu et al. 2017) | [Official](https://github.com/MarkWuNLP/MultiTurnResponseSelection) | 0.7327 |	0.927 |	0.726	|0.847|	0.962 |[Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://www.aclweb.org/anthology/P17-1046.pdf)| Multi-turn|
|DAM(Zhou et al. 2018) | [Official](https://github.com/baidu/Dialogue/tree/master/DAM) | \ |	0.938 |	0.767 |	0.874 |	0.969 |[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf)  | multi-turn|
|DUA (Zhang et al. 2018)|[Official](https://github.com/cooelf/DeepUtteranceAggregation)| \ |	\ |	0.752 |	0.868 |	0.962 |[Modeling Multi-turn Conversation with Deep Utterance Aggregation](https://arxiv.org/pdf/1806.09102.pdf)|multi-turn|
| DMN (Yang et al. 2018)| [Official](https://github.com/yangliuy/NeuralResponseRanking) | 0.7719 | \ | \ |	\ |	\ |[Response Ranking with Deep Matching Networks and External Knowledge in Information-seeking Conversation Systems](https://arxiv.org/pdf/1805.00188.pdf) |multi-turn |
|U2U-IMN(Gu et al. 2019 a)|[Official](https://github.com/JasonForJoy/U2U-IMN) | 0.866 |	0.945 |	0.790 |	0.886 |	0.973 |[Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1911.06940v1.pdf)|multi-turn|
|TripleNet(Ma et al. 2019)|[Official](https://github.com/wtma/TripleNet.)|\ |	0.943	| 0.79	| 0.885 |	0.97 |[TripleNet: Triple Attention Network for Multi-Turn Response Selection in Retrieval-based Chatbots](https://arxiv.org/pdf/1909.10666v2.pdf)|multi-turn|
|IMN(Gu et al. 2019 b)|[Official](https://github.com/JasonForJoy/IMN)| \ |	0.946 |	0.794 |	0.889 |	0.974|[Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1901.01824v2.pdf)|multi-turn|
|IOI-local(Tao et al. 2019)|[Official](https://github.com/chongyangtao/IOI)| \ | 0.947 | 	0.796 | 	0.894 | 	0.974 | [One Time of Interaction May Not Be Enough: Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues](https://www.aclweb.org/anthology/P19-1001.pdf)|multi-turn|
|MSN(Yuan et al. 2019)|[Official](https://github.com/chunyuanY/Dialogue)|\ |\ |	0.8 |	0.899 |	0.978 | [Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots](https://www.aclweb.org/anthology/D19-1011.pdf)|multi-turn|
|SA-BERT(Gu et al. 2020)|[Official](https://github.com/JasonForJoy/SA-BERT)| \ |	0.965 |	0.855 |	0.928 |	0.983 |[Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/2004.03588v1.pdf)|multi-turn|
Douban Conversation Corpus					
| Model   | Code | MAP | MRR | P@1|	$R_{10}@1$ |$R_{10}@2$ |$R_{10}@5$|Paper| type |
|  ----  | ----  |  ---- | ---- | ---- | ---- | ---- | ---- |  ----  | ----  |
| Multi-View (Zhou et al. 2016)|  | 0.505 |	0.543 |	0.342 |	0.202 |	0.350 |	0.729 |  [Multi-view Response Selection for Human-Computer Conversation](https://www.aclweb.org/anthology/D16-1036.pdf) | multi-turn |
| DL2R (Yan, Song and Wu 2016)| |0.488 |	0.527 |	0.33 |	0.193 |	0.342 |	0.705 |[Learning to Respond with Deep Neural Networks for Retrieval-Based Human-Computer Conversation System](http://www.ruiyan.me/pubs/SIGIR2016.pdf) | multi-turn|
| SMN (Wu et al. 2017) | [Official](https://github.com/MarkWuNLP/MultiTurnResponseSelection)| 0.529 |	0.572 |	0.397 |	0.236 |	0.396 |	0.734 |[Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://www.aclweb.org/anthology/P17-1046.pdf)| Multi-turn|
|DAM(Zhou et al. 2018) | [Official](https://github.com/baidu/Dialogue/tree/master/DAM) | 0.55 |	0.601 |	0.427 |	0.254 |	0.410 |	0.757 |[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf)  | multi-turn|
|DUA (Zhang et al. 2018)|[Official](https://github.com/cooelf/DeepUtteranceAggregation)| 0.551 |	0.599 |	0.421 |	0.243 |	0.421 |	0.780 |[Modeling Multi-turn Conversation with Deep Utterance Aggregation](https://arxiv.org/pdf/1806.09102.pdf)|multi-turn|
|U2U-IMN(Gu et al. 2019 a)|[Official](https://github.com/JasonForJoy/U2U-IMN) |0.564 | 0.611 |	0.429 |	0.259 |	0.43 |	0.791 |[Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1911.06940v1.pdf)|multi-turn|
|TripleNet(Ma et al. 2019)|[Official](https://github.com/wtma/TripleNet.)|0.564 | 0.618 |	0.447 |	0.268 |	0.426 |	0.778 |[TripleNet: Triple Attention Network for Multi-Turn Response Selection in Retrieval-based Chatbots](https://arxiv.org/pdf/1909.10666v2.pdf)|multi-turn|
|IMN(Gu et al. 2019 b)|[Official](https://github.com/JasonForJoy/IMN)|0.570 | 0.615 |	0.433 |	0.262 |	0.452 |	0.789 |[Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1901.01824v2.pdf)|multi-turn|
|IOI-local(Tao et al. 2019)|[Official](https://github.com/chongyangtao/IOI)| 0.573 | 0.621 |	0.444 |	0.269 |	0.451 |	0.786 |[One Time of Interaction May Not Be Enough: Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues](https://www.aclweb.org/anthology/P19-1001.pdf)|multi-turn|
|MSN(Yuan et al. 2019)|[Official](https://github.com/chunyuanY/Dialogue)| 0.587 | 0.632 |	0.470 |	0.295 |	0.452 |	0.788 |[Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots](https://www.aclweb.org/anthology/D19-1011.pdf)|multi-turn|
|SA-BERT(Gu et al. 2020)|[Official](https://github.com/JasonForJoy/SA-BERT)| 0.619 | 0.659 |	0.496 |	0.313 |	0.481 |	0.847 |[Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/2004.03588v1.pdf)|multi-turn|
MSDialog			
| Model   | Code | MAP |	Recall@5|	Recall@1| Recall@2|Paper| type |
|  ----  | ----  |  ---- | ---- | ---- | ---- |  ----  | ----  |
| DMN (Yang et al. 2018)|[Official](https://github.com/yangliuy/NeuralResponseRanking)| 0.6792 |	0.9356 |	0.5021 |	0.7122 |[Response Ranking with Deep Matching Networks and External Knowledge in Information-seeking Conversation Systems](https://arxiv.org/pdf/1805.00188.pdf) |multi-turn |
E-commerce Corpus			
| Model   | Code | MAP |	$R_{10}@1$ |	$R_{10}@2$ |	$R_{10}@5$ | Paper| type |
|  ----  | ----  |  ---- | ---- | ---- | ---- | ----  | ----  |
| Multi-View (Zhou et al. 2016)| | \ |	0.421 |	0.601 |	0.861 |  [Multi-view Response Selection for Human-Computer Conversation](https://www.aclweb.org/anthology/D16-1036.pdf) | multi-turn |
| DL2R (Yan, Song and Wu 2016)| | \	 |0.399 | 	0.571 |	0.842 |[Learning to Respond with Deep Neural Networks for Retrieval-Based Human-Computer Conversation System](http://www.ruiyan.me/pubs/SIGIR2016.pdf) | multi-turn|
| SMN (Wu et al. 2017) | [Official](https://github.com/MarkWuNLP/MultiTurnResponseSelection)| \ | 	0.453 |	0.654 |	0.886 | [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://www.aclweb.org/anthology/P17-1046.pdf)| Multi-turn|
|DAM(Zhou et al. 2018) | [Official](https://github.com/baidu/Dialogue/tree/master/DAM) | \ |	0.526 |	0.727 |	0.933 |[Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://www.aclweb.org/anthology/P18-1103.pdf)  | multi-turn|
|DUA (Zhang et al. 2018)|[Official](https://github.com/cooelf/DeepUtteranceAggregation)| \	| 0.501 |	0.700 |	0.921 |[Modeling Multi-turn Conversation with Deep Utterance Aggregation](https://arxiv.org/pdf/1806.09102.pdf)|multi-turn|
|U2U-IMN(Gu et al. 2019 a)|[Official](https://github.com/JasonForJoy/U2U-IMN) |0.759 |	0.616 |	0.806 |	0.966 |[Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1911.06940v1.pdf)|multi-turn|
|IMN(Gu et al. 2019 b)|[Official](https://github.com/JasonForJoy/IMN)| \ |	0.621 |	0.797 |	0.964 |[Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1901.01824v2.pdf)|multi-turn|
|IOI-local(Tao et al. 2019)|[Official](https://github.com/chongyangtao/IOI)|\ |	0.563 |	0.768 |	0.950 |[One Time of Interaction May Not Be Enough: Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues](https://www.aclweb.org/anthology/P19-1001.pdf)|multi-turn|
|MSN(Yuan et al. 2019)|[Official](https://github.com/chunyuanY/Dialogue)| \ |	0.606 |	0.770 |	0.937 |[Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots](https://www.aclweb.org/anthology/D19-1011.pdf)|multi-turn|
|SA-BERT(Gu et al. 2020)|[Official](https://github.com/JasonForJoy/SA-BERT)| \ |	0.704 |	0.879 |	0.985 |[Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/2004.03588v1.pdf)|multi-turn|
