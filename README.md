# NER-CONLL2003

###什么是NER？

命名实体识别（NER）是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。命名实体识别是信息提取、问答系统、句法分析、机器翻译等应用领域的重要基础工具，作为结构化信息提取的重要步骤。摘自BosonNLP

##Task CoNLL 2003

[CoNLL-2003 Shared Task](https://cogcomp.org/page/resource_view/81): Language-Independent Named Entity Recognition

The CoNLL-2003 (Sang et al. 2003) shared task deals with language-independent named entity recognition as well (English and German).

## Results

| References                               | Method                                   | F1    |
| ---------------------------------------- | ---------------------------------------- | ----- |
| [Ma 2016](https://arxiv.org/pdf/1603.01354.pdf) | CNN-bidirectional LSTM-CRF               | 91.21 |
| Luo et al. (2015)                        | JERL                                     | 91.20 |
| Chiu et al. (2015)                       | BLSTM-CNN + emb + lex                    | 91.62 |
| Huang et al. (2015)                      | BI-LSTM-CRF                              | 90.10 |
| Passos et al. (2014)                     | Baseline + Gaz + LexEmb                  | 90.90 |
| Suzuki et al. (2011)                     | L1CRF                                    | 91.02 |
| Collobert et al. (2011)                  | NN+SLL+LM2+Gazetteer                     | 89.59 |
| Collobert et al. (2011)                  | NN+SLL+LM2                               | 88.67 |
| Ratinov et al. (2009)                    | Word-class Model                         | 90.80 |
| Lin et al. (2009)                        | W500 + P125 + P64                        | 90.90 |
| Ando et al. (2005)                       | Semi-supervised approach                 | 89.31 |
| Florian et al. (2003)                    | Combination of various machine-learning classifiers | 88.76 |

---

###模型训练过程

1. 首先导入数据 training，validation，test

2. 把单词转化成 one-hot 向量后，再转化成词向量

3. 输入层的输入，以每个词为中心,取其窗口大小为3的上下文语境

   L2 正则化和用 dropout 来减小过拟合

   Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

4. 用交叉熵来计算误差

5. J 对各个参数进行求导

   Adam优化算法更新梯度，不断地迭代，使得loss越来越小直至收敛。



## References

- **End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF** (ACL'16), Ma et al. [[pdf](https://arxiv.org/pdf/1603.01354.pdf)]
- **Named Entity Recognition with Bidirectional LSTM-CNNs** (CL'15), JPC Chiu et al. [[pdf](https://arxiv.org/pdf/1511.08308.pdf)]
- **Bidirectional LSTM-CRF Models for Sequence Tagging** (EMNLP'15), Z Huang et al. [[pdf](https://arxiv.org/pdf/1508.01991.pdf)]
- **Joint entity recognition and disambiguation** (EMNLP '15), G Luo et al. [[pdf](http://aclweb.org/anthology/D15-1104)]
- **Lexicon infused phrase embeddings for named entity resolution** (ACL'14), A Passos et al. [[pdf](http://www.aclweb.org/anthology/W14-1609)]
- **Learning condensed feature representations from large unsupervised data sets for supervised learning** (ACL'11), J Suzuki et al. [[pdf](http://www.aclweb.org/anthology/P11-2112)]
- **Natural Language Processing (Almost) from Scratch** (CL'11), R Collobert et al. [[pdf](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)]
- **Design Challenges and Misconceptions in Named Entity Recognition** (CoNLL'09), L Ratinov et al. [[pdf](http://www.aclweb.org/anthology/W09-1119)]
- **Phrase Clustering for Discriminative Learning** (ACL '09), D Lin et al. [[pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35520.pdf)]
- **A Framework for Learning Predictive Structures from Multiple Tasks and Unlabeled Data** (JMLR'05), RK Ando et al. [[pdf](http://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf)]
- **Named Entity Recognition through Classifier Combination** (HLT-NAACL'03), R Florian et al. [[pdf](http://clair.si.umich.edu/clair/HLT-NAACL03/conll/pdf/florian.pdf)]
- **Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition** (CoNLL'03), EFTK Sang et al. [[pdf](http://aclweb.org/anthology/W03-0419)]

**See Also**

- [☶ Named Entity Recognition (State of The Art)](https://github.com/magizbox/underthesea/wiki/English-NLP-SOTA#named-entity-recognition)




