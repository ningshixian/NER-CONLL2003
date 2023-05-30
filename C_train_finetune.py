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
from bert4keras.optimizers import Adam, AdaFactor
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField

import tensorflow as tf
from keras.optimizers import RMSprop, Nadam
from keras.layers import Input, Embedding, Dense, LSTM, Dropout, CuDNNLSTM, Conv1D, add
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, model_from_json, load_model
from keras.regularizers import l2
from keras.backend.tensorflow_backend import set_session

# from B_model import NNModel
from utils import load_data, data_generator, produce_matrix
from config import Config


np.random.seed(1671)  # 随机数种子，重复性设置
# 混合精度训练 参考《模型训练太慢？显存不够用？这个算法让你的GPU老树开新花》
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE"] = "1"

session_conf = tf.ConfigProto()
session_conf.allow_soft_placement = True
session_conf.log_device_placement = False
session_conf.gpu_options.allow_growth = True  # 按需求增长
# session_conf.gpu_options.per_process_gpu_memory_fraction=0.3    # 按比例固定
session = tf.Session(config=session_conf)
set_session(session)

config = Config()
print("Model Type: ", config.model_type)
print("Fine Tune Learning Rate: ", config.learning_rate)
print("Data dir: ", config.processed_data_dir)
print("Pretrained Model Vocab: ", config.dict_path)
print("bilstm embedding ", config.lstm_dim)
print("use bert: ", config.pretrained_model_type)

CRF = ConditionalRandomField(lr_multiplier=config.crf_learning_rate)

# 标注数据
train_data = load_data(config.train_data, config.nb_sentence)
valid_data = load_data(config.dev_data, config.nb_sentence)
test_data = load_data(config.test_data, config.nb_sentence)
print("数据长度(句子数量) train: {} valid: {} test: {}\n".format(len(train_data), len(valid_data), len(test_data)))
# train: 286579 valid: 15922 test: 15922

# 建立分词器
tokenizer = Tokenizer(config.dict_path, do_lower_case=True)

# label类别映射
labels = ["person", "city", "project", "time"]  # 4种类型
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1
print("NB_CLASSES: ", num_labels)  # 9
print("label2id ", label2id)


class NNModel:
    def __init__(self, config, NB_CLASSES, embedding_matrix=None):
        self.config = config
        self.NB_CLASSES = NB_CLASSES
        # 超参数设置
        self.lstm_dim = self.config.lstm_dim
        self.l2_rate = self.config.l2_rate
        self.embed_dense_dim = self.config.embed_dense_dim
        self.hidden_dim = self.config.hidden_dim
        self.dropout = self.config.dropout
        self.model_type = self.config.model_type
        self.pretrained_model_type = self.config.pretrained_model_type
        self.embedding_matrix = "" if self.pretrained_model_type else embedding_matrix
        self.sequence_length = self.config.sequence_length
        self.optimizer = self.config.optimizer
        self.learning_rate = self.config.learning_rate
        self.save_model_path = self.config.save_model_architecture
        self.bert_layers = self.config.bert_layers
        self.config_path = self.config.config_path
        self.checkpoint_path = self.config.checkpoint_path

    def init_embedding(self):
        """
        词向量层初始化
        :param bert_init:
        :return:
        """
        tokens_input = Input(shape=(self.sequence_length,), name="tokens_input", dtype="int32")
        emb_layer = Embedding(
            input_dim=self.embedding_matrix.shape[0],  # 索引字典大小
            output_dim=self.embedding_matrix.shape[1],  # 词向量的维度
            weights=[self.embedding_matrix],
            trainable=True,
            # mask_zero=True,    # 若√则编译报错，CuDNNLSTM 不支持？
            name="token_emd",
        )
        tokens_emb = emb_layer(tokens_input)
        return tokens_input, tokens_emb

    def bert_embed(self, bert_init=True):
        """
        读取BERT的TF模型 + 对BERT的Embedding降维
        :param bert_init:
        :return:
        """
        if self.pretrained_model_type == "albert":
            _model = build_transformer_model(self.config_path, self.checkpoint_path, model="albert")
            output_layer = "Transformer-FeedForward-Norm"
            output = _model.get_layer(output_layer).get_output_at(self.bert_layers - 1)
        else:
            _model = build_transformer_model(self.config_path, self.checkpoint_path, model="bert")
            output_layer = "Transformer-%s-FeedForward-Norm" % (self.bert_layers - 1)
            output = _model.get_layer(output_layer).output
        # output = Dense(output.output_shape[-1], activation="relu")(output)
        return _model.input, output

    def biLSTM_layer(self, tokens_emb):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        if self.model_type == "bilstm":
            lstm = Bidirectional(
                LSTM(
                    units=self.lstm_dim,
                    return_sequences=True,
                    # kernel_regularizer=self.l2_rate,
                    # bias_regularizer=self.l2_rate,
                )
            )
        elif self.model_type == "bilstm-gpu":
            lstm = Bidirectional(
                CuDNNLSTM(
                    units=self.lstm_dim,
                    return_sequences=True,
                    # kernel_regularizer=self.l2_rate,
                    # bias_regularizer=self.l2_rate,
                )
            )
        else:
            raise KeyError

        output = lstm(tokens_emb)
        return output

    def IDCNN_layer(self, tokens_emb):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        
        TypeError:Layer conv1d_1 does not support masking
        activation="relu"
        """
        x = Conv1D(filters=256, kernel_size=2, padding="same", dilation_rate=1)(tokens_emb)
        x = Conv1D(filters=256, kernel_size=3, padding="same", dilation_rate=1)(x)
        x = Conv1D(filters=512, kernel_size=4, padding="same", dilation_rate=2)(x)
        return x

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        output = TimeDistributed(Dense(self.hidden_dim, activation="relu"))(lstm_outputs)  # 可加可不加
        output = Dropout(self.dropout)(output)
        output = TimeDistributed(Dense(self.NB_CLASSES, activation="relu", name="final_layer"))(
            output
        )  # 不加激活函数，否则预测结果有问题222222 - activation="relu"
        return output

    def buildModel(self):
        if self.pretrained_model_type:
            tokens_input, tokens_emb = self.bert_embed()
        else:
            tokens_input, tokens_emb = self.init_embedding()

        tokens_emb = Dropout(self.dropout)(tokens_emb)

        if "bilstm" in self.model_type:
            output = self.biLSTM_layer(tokens_emb)
        elif "idcnn" in self.model_type:
            output = self.IDCNN_layer(tokens_emb)
        elif "attention" in self.model_type:
            output = AttentionSelf(256)(tokens_emb)
        else:
            output = tokens_emb

        output = self.project_layer(output)
        # crf = ChainCRF(name="CRF")
        # output = crf(output)
        output = CRF(output)
        model = Model(inputs=tokens_input, outputs=output)
        model.summary()

        if self.optimizer.lower() == "adam":
            opt = Adam(lr=self.learning_rate)
        elif self.optimizer.lower() == "nadam":
            opt = Nadam(lr=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            opt = RMSprop(lr=self.learning_rate)

        # if self.config.pretrained_model_type:
        #     # 混合精度：需要对优化器(Optimizer)作如下修改
        #     opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        #     opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
        #         opt, loss_scale="dynamic"
        #     )  # 需要添加这句话，该例子是tf1.14.0版本,不同版本可能不一样

        # model.compile(loss=crf.sparse_loss, optimizer=opt, metrics=["accuracy"])
        model.compile(loss=CRF.sparse_loss, optimizer=opt, metrics=[CRF.sparse_accuracy])
        # 保存模型的结构
        json_model = model.to_json()
        with open(self.config.save_model_architecture, "w") as f:
            f.write(json_model)
        # plot_model(model, to_file='result/model.png', show_shapes=True)
        return model


class Ner:
    """封装模型的训练和预测方法。
    """

    def __init__(self, NB_CLASSES, emb_matrix, reload=False):
        # 读取模型结构图
        if reload:
            print("model reconstruction from JSON...")
            with open(config.save_model_architecture) as f:
                self.model = model_from_json(f.read())
                # obj = ConditionalRandomField.create_custom_objects()
                # model = model_from_json(f.read(), custom_objects=obj)
            self.model.load_weights(config.save_model_weights)
        else:
            M = NNModel(config, NB_CLASSES, emb_matrix)
            self.model = M.buildModel()

    def train(self, train_generator):
        # callback设置
        evaluator = Evaluate()
        # 模型训练
        self.model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=config.train_epoch,
            callbacks=[evaluator],
        )
        # model.fit(
        #     x=data_ids,
        #     y=label_ids,
        #     epochs=config.train_epoch,
        #     batch_size=config.batch_size,
        #     shuffle=True,
        #     callbacks=[saveModel],
        #     validation_split=0.1,
        # )

    def predict(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        if config.pretrained_model_type:
            nodes = self.model.predict([[token_ids], [segment_ids]])[0]
            # nodes = self.model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
        else:
            token_ids = sequence_padding([token_ids], length=config.sequence_length)
            nodes = self.model.predict(token_ids)[0]
        # print([np.argmax(item) for item in nodes])
        # self.CRF = self.model.get_layer(index=-1)  # "crf_layer"
        trans = K.eval(CRF.trans)
        labels = viterbi_decode(nodes, trans)
        # print(labels)
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
            try:
                if mapping[w[0]] and mapping[w[-1]]:  # 排除最后的标志位被识别为实体的情况
                    xx.append((text[mapping[w[0]][0] : mapping[w[-1]][-1] + 1], l))
            except Exception as e:
                # "非bert模型padding的部分报 IndexError！"
                pass
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
        # print([tuple(i) for i in d if i[1] != "O"])
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
        if epoch > 1:
            trans = K.eval(CRF.trans)
            # print(trans)
            f1, precision, recall = evaluate(valid_data)
            # 保存最优
            if f1 >= self.best_val_f1:
                self.best_val_f1 = f1
                ner.model.save_weights(config.save_model_weights)  # 保存模型的权重
            print(
                "valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n"
                % (f1, precision, recall, self.best_val_f1)
            )
            f1, precision, recall = evaluate(test_data)
            print("test:  f1: %.5f, precision: %.5f, recall: %.5f\n" % (f1, precision, recall))


def test_model(text):
    """加载预训练的模型权重&预测
    """
    ner = Ner(num_labels, emb_matrix, reload=True)
    CRF = ner.model.get_layer(index=-1)  # "crf_layer"
    return ner.predict(text)


if __name__ == "__main__":

    # 输入数据生成
    print("done! Model building....")
    train_generator = data_generator(label2id, train_data, tokenizer)  # ((64,64),64)
    vocab_size = tokenizer._vocab_size  # 21128
    token_to_id = tokenizer._token_dict
    emb_matrix = produce_matrix(token_to_id, config.embeddingFile)  # 词向量矩阵生成及保存

    # 模型训练
    ner = Ner(num_labels, emb_matrix)
    ner.train(train_generator)

    # # 预测
    # text = """
    #     哦哦哦，我姓王，叫王灏（微信号Sir_li_），在集团总部工作，住在北京景粼原著, 在地产航道，已婚，现住在郦湾，职级是p5,
    #     电话134569675439，还想报销电脑、酒水和绿植，然后换钱来买探亲机票，然后生病有事请假，抵扣联，对继续等待，
    #     另外帮忙查下bx20190724231这单谁审的。我准备在明天下午4点左右看房（月租是6300/月）
    #     """
    # print(test_model(text))
