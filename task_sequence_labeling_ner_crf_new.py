#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.18%，测试集的F1可以到95.35%

import os, sys
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField

import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM, Conv1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, model_from_json, load_model
from keras.regularizers import l2


np.random.seed(1671)  # 随机数种子，重复性设置
# 混合精度训练 参考《模型训练太慢？显存不够用？这个算法让你的GPU老树开新花》
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE"] = "1"
local_or_remote = sys.argv[1]

# set GPU memory
if local_or_remote != "local":

    # 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 按需求增长
    sess = tf.Session(config=config)
    set_session(sess)

    # # 方法2:只允许使用x%的显存,其余的放着不动
    # from keras.backend.tensorflow_backend import set_session
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5    # 按比例
    # sess = tf.Session(config=config)

    # # 方法3: 针对tf2.0 https://blog.csdn.net/vera425/article/details/103261137
    # gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"


maxlen = 150  # 256
epochs = 20
batch_size = 64  # 32
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)

# albert配置
if local_or_remote == "local":
    config_path = "D:/#Pre-trained_Language_Model/weights/albert/albert_large/albert_config.json"
    checkpoint_path = "D:/#Pre-trained_Language_Model/weights/albert/albert_large/model.ckpt"
    dict_path = "D:/#Pre-trained_Language_Model/weights/albert/albert_large/vocab_chinese.txt"
    nb_sentence = 64
else:
    config_path = "/data01/ningshixian/bert/albert_large/albert_config.json"
    checkpoint_path = "/data01/ningshixian/bert/albert_large/model.ckpt"
    dict_path = "/data01/ningshixian/bert/albert_large/vocab_chinese.txt"
    nb_sentence = -1


def load_data(filename):
    D = []
    with open(filename, encoding="utf-8") as f:
        f = f.read()
        for l in f.split("\n\n")[:nb_sentence]:
            if not l:
                continue
            d, last_flag = [], ""
            for c in l.split("\n"):
                if not c.strip():
                    continue
                char, this_flag = c.split("\t")
                # char, this_flag = c.split(" ")
                if this_flag == "O" and last_flag == "O":
                    d[-1][0] += char
                elif this_flag == "O" and last_flag != "O":
                    d.append([char, "O"])
                elif this_flag[:1] == "B":
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 标注数据
train_data = load_data("datasets/longfor-ner-corpus/example.train")
valid_data = load_data("datasets/longfor-ner-corpus/example.dev")
test_data = load_data("datasets/longfor-ner-corpus/example.test")
print("数据长度 train: {} valid: {} test: {}\n".format(len(train_data), len(valid_data), len(test_data)))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ["person", "city", "project", "time"]  # 4种类型
# labels = ["PER", "LOC", "ORG"]
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    定义 Dataset 类，封装一些数据读入和预处理方法
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == "O":
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += [B] + [I] * (len(w_token_ids) - 1)
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class Ner:
    """定义一个 BiLSTM+CRF 类，封装模型的构建，训练和预测方法。
    """

    def __init__(self, load_flag=False):
        if not load_flag:
            self.model = self.build_model()
        else:
            print("model reconstruction from JSON...")
            with open("models/model_cpu.json") as f:
                # obj = ConditionalRandomField.create_custom_objects()
                # model = model_from_json(f.read(), custom_objects=obj)
                self.model = model_from_json(f.read())

    def build_model(self):
        """
        后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：

        model = build_transformer_model(
            config_path,
            checkpoint_path,
            model='albert',
        )

        output_layer = 'Transformer-FeedForward-Norm'
        output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
        """

        lstm_cpu = Bidirectional(LSTM(units=128, return_sequences=True, activation="tanh"))
        lstm_gpu = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))
        cnn = Conv1D(filters=128, kernel_size=3, activation="relu")
        encoder = lstm_cpu

        # (model='bert'/'albert'/'nezha'/'electra'/'gpt2_ml'/'t5')

        # model = build_transformer_model(config_path, checkpoint_path, model="bert")
        # output_layer = "Transformer-%s-FeedForward-Norm" % (bert_layers - 1)
        # output = model.get_layer(output_layer).output
        model = build_transformer_model(config_path, checkpoint_path, model="albert")
        output_layer = "Transformer-FeedForward-Norm"
        output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
        output = Dropout(0.5)(output)
        output = encoder(output)
        output = TimeDistributed(Dense(128, activation="relu"))(output)
        output = Dropout(0.5)(output)
        output = TimeDistributed(Dense(num_labels, activation="relu"))(output)
        output = CRF(output)
        model = Model(model.input, output)
        # model.summary()
        # model.compile(loss=CRF.sparse_loss, optimizer=Adam(learing_rate), metrics=[CRF.sparse_accuracy])

        opt = tf.compat.v1.train.AdamOptimizer(learing_rate)
        opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
            opt, loss_scale="dynamic"
        )  # 需要添加这句话，该例子是tf1.14.0版本,不同版本可能不一样
        model.compile(loss=CRF.sparse_loss, optimizer=opt, metrics=[CRF.sparse_accuracy])
        return model

    def train(self, train_generator, epochs, evaluator):
        self.model.fit_generator(
            train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator]
        )
        # 保存模型的结构
        json_model = self.model.to_json()
        with open("models/model_cpu.json", "w") as f:
            f.write(json_model)

    def predict(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = self.model.predict([[token_ids], [segment_ids]])[0]
        trans = K.eval(CRF.trans)
        labels = viterbi_decode(nodes, trans)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        # result_label = [np.argmax(i) for i in result]
        # return result_label

        xx = []
        for w, l in entities:
            if mapping[w[0]] and mapping[w[-1]]:  # 排除最后的标志位被识别为实体的情况
                xx.append((text[mapping[w[0]][0] : mapping[w[-1]][-1] + 1], l))
        return xx
        # return [(text[mapping[w[0]][0] : mapping[w[-1]][-1] + 1], l) for w, l in entities if mapping[w[0]] and mapping[w[-1]]]


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = "".join([i[0] for i in d])
        R = set(ner.predict(text))
        # print(R)
        T = set([tuple(i) for i in d if i[1] != "O"])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 10:
            trans = K.eval(CRF.trans)
            # print(trans)
            f1, precision, recall = evaluate(valid_data)
            # 保存最优
            if f1 >= self.best_val_f1:
                self.best_val_f1 = f1
                ner.model.save_weights("models/best_model.weights")
            print(
                "valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n"
                % (f1, precision, recall, self.best_val_f1)
            )
            # f1, precision, recall = evaluate(test_data)
            # print("test:  f1: %.5f, precision: %.5f, recall: %.5f\n" % (f1, precision, recall))


def test_model(text):
    """加载预训练的模型权重&预测
    """
    ner = Ner()
    filepath = "models/albert_large_lstm_crf/best_model.weights"
    ner.model.load_weights(filepath)
    print("加载模型成功!!")
    return ner.predict(text)


if __name__ == "__main__":

    train_generator = data_generator(train_data, batch_size)

    evaluator = Evaluate()
    ner = Ner()
    ner.train(train_generator, epochs, evaluator)

    # 预测
    text = """
        哦哦哦，我姓王，叫王灏（微信号Sir_li_），在集团总部工作，住在北京景粼原著, 在地产航道，已婚，现住在郦湾，职级是p5,
        电话134569675439，还想报销电脑、酒水和绿植，然后换钱来买探亲机票，然后生病有事请假，抵扣联，对继续等待，
        另外帮忙查下bx20190724231这单谁审的。我准备在明天下午4点左右看房（月租是6300/月）
        """
    print(test_model(text))
