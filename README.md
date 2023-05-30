# bert-based-ner

## 实验环境

部署43服务器，nsx_env环境，GPU (Tesla P100 16G)

## 依赖包安装

```
python3 -m pip install -r requirements.txt
```

## 实验流程

> 代码结构

1. 数据预处理 preprocess.py

```
利用正则定位噪声，对原始数据的清洗
清除非中英文和非常见标点符号的字符
......
标注4种类型的实体
采用BIO标注策略
修改longfor数据的处理代码bug
去除paddleLAC提取出的数字时间
去除无标注实体的句子
训练集/验证集/测试集划分
将清洗后的数据转换成CONLL格式

数据量2342*32
```

2. 构建实体识别模型 model.py

   尝试使用多种预训练语言模型（BERT/AL_BERT/ROBERT），并分别下接了IDCNN-CRF和BLSTM-CRF两种结构来构建实体识别模型。


3. 工具类 utils.py

   数据迭代器，用于生成batch数据喂给模型


4. 模型训练 train_finetune.py

   - 数据的喂入
   - 模型的fine-tuning
   - 模型保存

5. 模型预测 predict.py

   - 读取最优F1结果的模型，对测试集进行预测
   - 将生成的概率文件复原成文字结果

6. post_process.py

   后处理脚本

7. config.py

   超参数设置和路径设置


## 实验结果

1. BLSTM+CRF

  test: f1: 0.82073, precision: 0.70942, recall: 0.97346

2. BLSTM+CRF+BERT

   尝试加入 [`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) (版本chinese_L-12_H-768_A-12)，模型参数量过大，训练太慢，放弃;

3. BLSTM+CRF+ALBERT

   - 使用 [`albert-base, chinese`](https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz)，去掉BLSTM层
     - 迭代15次，用时11个小时
     - test: f1: 0.90598, precision: 0.92791, recall: 0.88505
   - 使用 [`albert-large, chinese`](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)，去掉BLSTM层
     - 迭代20次，用时12个小时，一次迭代需55分钟
     - test: f1: 0.91092, precision: 0.92695, recall: 0.89544
   - 尝试 [`albert-xlarge, chinese`](https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz)
     - 报OOM错误，内存不够，放弃！
   - 使用 [`albert-large, chinese`](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)，加入BLSTM层
     - 一次迭代需一个半小时
     - 调整max_len=150，batch_size=64后，一次迭代需50分钟左右，test一次1个半小时左右
     - test: f1: 0.89363, precision: 0.91806, recall: 0.87047（未迭代完....）
   - 继续改进
     1. 加入`自动混合精度训练` 
     2. 重计算技巧？AdaFactor？
     3. 迭代10次之后才开始计算在测试集上的PRF值
