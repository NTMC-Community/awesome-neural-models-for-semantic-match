<div align="center">
<img width="300" src="artworks/awesome.svg" alt="Awesome">
<br>
<br>
<p><b>Awesome Neural Models for Semantic Match</b></p>
</div>
<br>
<p align="center">
<sub>A collection of papers maintained by MatchZoo Team.</sub>
<br>
<sub>Checkout our open source toolkit <a href="https://github.com/faneshion/MatchZoo">MatchZoo</a> for more information!</sub>
</p>
<br>

Text matching is a core component in many natural language processing tasks, where many task can be viewed as a matching between two texts input.

<div align="center">
<img width="300" src="artworks/equation.svg" alt="equation">
</div>

Where **s** and **t** are source text input and target text input, respectively. The **psi** and **phi** are representation function for input **s** and **t**, respectively. The **f** is the interaction function, and **g** is the aggregation function. More detailed explaination about this formula can be found on [A Deep Look into Neural Ranking Models for Information Retrieval](https://arxiv.org/abs/1903.06902). The representative matching tasks are as follows:


| **Tasks** | **Source Text**   | **Target Text**  |
| :-------: | :----------------: | :--------------: |
| [Ad-hoc Information Retrieval](Ad-hoc Information Retrieval/Ad-hoc Information Retrieval.md)| query   | document (title/content) |
| [Community Question Answer](Community Question Answer/Community Question Answer.md)   | question| question/answer          |
| [Paraphrase Indentification](Paraphrase Identification/Paraphrase Identification.md)  | string1 | string2                  |
| [Natural Language Inference](Natural Language Inference/Natural Language Inference.md)  | premise | hypothesis               |

### Healthcheck

```python
pip3 install -r requirements.txt
python3 healthcheck.py
```