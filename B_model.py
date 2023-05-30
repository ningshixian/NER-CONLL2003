import codecs
import time
import pickle as pkl
from tqdm import tqdm
import numpy as np
from keras.callbacks import Callback
from keras.layers import Input, Embedding, Dense, LSTM, CuDNNLSTM, Dropout, Conv1D, add
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, model_from_json
from keras.optimizers import *
from keras.regularizers import l2, L1L2
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model, to_categorical
from keras.backend.tensorflow_backend import set_session

# from keraslayers.ChainCRF import ChainCRF
from keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from config import Config
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.optimizers import Adam, AdaFactorV1
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField


config = Config()
CRF = ConditionalRandomField(lr_multiplier=config.crf_learning_rate)


class AttentionSelf(Layer):
    """
        self attention,
        TypeError:Layer attention_self_1 does not support masking
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(
            name="WKV",
            shape=(3, input_shape[2], self.output_dim),
            initializer="uniform",
            regularizer=L1L2(0.0000032),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print("WQ.shape", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        print("QK.shape", QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


# 让Conv1D支持Masking
class MyConv1D(keras.layers.Conv1D):
    def __init__(self, *args, **kwargs):
        super(MyConv1D, self).__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]
        return super(MyConv1D, self).call(inputs)


class NNModel:
    def __init__(self, config, NB_CLASSES, embedding_matrix=None):
        self.config = config
        self.NB_CLASSES = NB_CLASSES
        # 超参数设置
        self.initializer = initializers.xavier_initializer()
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
                    activation="tanh"
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
        output = TimeDistributed(Dense(self.NB_CLASSES, name="final_layer"))(
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
            opt = Adam(lr=self.learning_rate, clipvalue=1.0)
        elif self.optimizer.lower() == "nadam":
            opt = Nadam(lr=self.learning_rate, clipvalue=1.0)
        elif self.optimizer.lower() == "rmsprop":
            opt = RMSprop(lr=self.learning_rate, clipvalue=1.0)

        if self.config.pretrained_model_type:
            # 混合精度：需要对优化器(Optimizer)作如下修改
            opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                opt, loss_scale="dynamic"
            )  # 需要添加这句话，该例子是tf1.14.0版本,不同版本可能不一样

        # model.compile(loss=crf.sparse_loss, optimizer=opt, metrics=["accuracy"])
        model.compile(loss=CRF.sparse_loss, optimizer=opt, metrics=[CRF.sparse_accuracy])
        # 保存模型的结构
        json_model = model.to_json()
        with open(self.config.save_model_architecture, "w") as f:
            f.write(json_model)
        # plot_model(model, to_file='result/model.png', show_shapes=True)
        return model


if __name__ == "__main__":

    # =========================模型构建============================ #

    print("done! Model building....")
    embedding_matrix = np.asarray([[0.2] * 20, [0.1] * 20])
    saveModel = ModelCheckpoint(
        filepath=config.checkpoint_path,
        monitor="val_loss",
        save_best_only=True,  # 只保存在验证集上性能最好的模型
        save_weights_only=True,  # 只保存模型权重
        mode="auto",
    )
    earlyStop = EarlyStopping(monitor="val_loss", patience=5, mode="auto")
    tensorBoard = TensorBoard(log_dir="./model", histogram_freq=0)  # 计算各个层激活值直方图的频率(每多少个epoch计算一次)
    # calculatePRF1 = ConllevalCallback(test_x, test_y, 0, id_to_tag, sentence_maxlen, 0, batch_size)

    # 读取模型结构图
    NB_CLASSES = 9
    M = NNModel(config, NB_CLASSES, embedding_matrix)
    model = M.buildModel()
