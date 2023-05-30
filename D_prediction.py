import os
import numpy as np
from keras.models import load_model
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm

# from keraslayers.ChainCRF import ChainCRF, create_custom_objects
import codecs
from sklearn.metrics import classification_report
from config import Config
from B_model import NNModel
from keras.models import model_from_json
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.backend import keras, K
from utils import load_data, data_generator, produce_matrix
from C_train_finetune import Ner


config = Config()
print("Model Type: ", config.model_type)
print("Fine Tune Learning Rate: ", config.embed_learning_rate)
print("Data dir: ", config.processed_data_dir)
print("Pretrained Model Vocab: ", config.dict_path)
print("bilstm embedding ", config.lstm_dim)
print("use bert: ", config.pretrained_model_type)

tokenizer = Tokenizer(config.dict_path, do_lower_case=True)
# 类别映射
labels = ["person", "city", "project", "time"]  # 4种类型
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1
print("NB_CLASSES: ", num_labels)  # 9


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
        # print(ner.predict(text))
        print([tuple(i) for i in d if i[1] != "O"])
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def prediction2label(pred, id_to_tag):
    """将预测结果转换为标签序列
    """
    out = []
    for p in pred:
        out.append(id_to_tag[p])
    return out


def iob_ranges(words, tags):
    """从预测的标签序列中抽取实体 IOB -> Ranges
    """
    assert len(words) == len(tags)
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split("_")[0] == "O":
            ranges.append({"entity": words[begin : i + 1], "type": temp_type, "start": begin, "end": i})

    for i, tag in enumerate(tags):
        t = tag.split("_")[0]
        if t == "O":
            pass
        elif t == "B":
            begin = i
            temp_type = tag.split("_")[1]
            check_if_closing_range()
        elif t == "M":
            pass
        elif t == "E" or t == "I":  # 支持BIO和BMES两种标注策略
            check_if_closing_range()
    return ranges


def cal_prf_ner():
    """
    # 字符级PRF结果：(0.8207316739979783, 0.7094248009144241, 0.9734658958338734)
    # 实体级PRF结果：(0.9398149711314449, 0.939910177618694, 0.9397197839297773)
    """
    nb_sent = 0
    sent, label = [], []
    y_pred, y_true = [], []
    X, Y, Z = 1e-10, 1e-10, 1e-10
    with codecs.open("data/example.test", "r", "utf-8") as f:
        for item in tqdm(f):
            # if nb_sent == 50:
            #     break
            if not item == "\n":
                item = item.strip("\n").split("\t")
                sent.append(item[0])
                label.append(t2t[item[1]])
                # label.append(new_tag_to_id[item[1]])
            else:
                # 对测试数据进行预测
                s = [char_to_id.get(w, 0) for w in sent]
                s = s[:max_length] if len(s) > max_length else s + [0] * (max_length - len(s))
                data_ids = np.asarray([s])
                sent = sent[:max_length]  # 截断
                label = label[:max_length]  # 截断
                predictions = model.predict(data_ids, verbose=0)
                predictions = predictions.argmax(axis=-1)
                predictions = list(predictions[0][: len(sent)])
                # 根据预测的{(实体名, 类别, 位置)}进行PRF计算
                y_pred = prediction2label(predictions, id_to_tag)
                R = set([tuple(item.values()) for item in iob_ranges("".join(sent), y_pred)])
                T = set([tuple(item.values()) for item in iob_ranges("".join(sent), label)])
                X += len(R & T)
                Y += len(R)
                Z += len(T)
                sent, label = [], []

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


if __name__ == "__main__":

    with open("data/embedding_matrix.pkl", "rb") as f:
        emb_matrix = pkl.load(f)
    ner = Ner(num_labels, None, reload=True)

    # 测试单条数据
    text = """哦哦哦，我叫王灏，工作地点在集团总部（北京），住在北京景粼原著，
        电话134569675439，准备在明天下午4点左右看房（月租是6300/月）"""
    res = ner.predict(text)

    # 测试集PRF
    test_data = load_data(config.test_data, config.nb_sentence)
    f1, precision, recall = evaluate(test_data)
    print("test:  f1: %.5f, precision: %.5f, recall: %.5f\n" % (f1, precision, recall))

    ## ============================================================================ ##

    # print("Loading data...")
    # with open("data/dict.pkl", "rb") as f:
    #     char_to_id, id_to_char, tag_to_id, id_to_tag, max_length = pkl.load(f)
    # print(tag_to_id)
    # print(max_length)
    # t2t = {
    #     "O": "O",
    #     "B_time": "B_TIME",
    #     "I_time": "I_TIME",
    #     "B_project": "B_PROJ",
    #     "I_project": "I_PROJ",
    #     "B_person": "B_PER",
    #     "I_person": "I_PER",
    #     "B_city": "B_CITY",
    #     "I_city": "I_CITY",
    # }
    # old_tag_to_id = {
    #     "O": 0,
    #     "B_TIME": 1,
    #     "I_TIME": 2,
    #     "B_PROJ": 3,
    #     "I_PROJ": 4,
    #     "B_PER": 5,
    #     "I_PER": 6,
    #     "B_CITY": 7,
    #     "I_CITY": 8,
    # }
    # new_tag_to_id = {
    #     "O": 0,
    #     "B_time": 1,
    #     "I_time": 2,
    #     "B_project": 3,
    #     "I_project": 4,
    #     "B_person": 5,
    #     "I_person": 6,
    #     "B_city": 7,
    #     "I_city": 8,
    # }

    # words = """哦哦哦，我叫王灏，工作地点在集团总部（北京），住在北京景粼原著,
    #     电话134569675439，准备在明天下午4点左右看房（月租是6300/月）"""
    # s = [char_to_id.get(w, 0) for w in words]
    # if len(s) >= max_length:
    #     s = s[:max_length]
    # else:
    #     s = s + [0] * (max_length - len(s))
    # data_ids = np.asarray([s])

    # print("Loading model...")
    # with tf.device("/cpu:0"):
    #     # model reconstruction from JSON:
    #     from keras.models import model_from_json

    #     # model = load_model('model/weights1.15-11.90.hdf5', custom_objects=create_custom_objects())
    #     with open("model/model_cpu.json") as f:
    #         model = model_from_json(f.read(), custom_objects=create_custom_objects())
    #     filepath = "model/best_weights.hdf5"
    #     model.load_weights(filepath)
    #     print("加载模型成功!!")

    # # 预测
    # predictions = model.predict(data_ids, verbose=1)
    # y_pred = predictions.argmax(axis=-1)
    # y_pred = y_pred[0][: len(words)]

    # y_pred = prediction2label(list(y_pred), id_to_tag)
    # print("预测结果：", iob_ranges(words, y_pred))

    # """
    # 预测结果： [
    #     {'entity': '王灏', 'type': 'PER', 'start': 6, 'end': 7},
    #     {'entity': '北京', 'type': 'CITY', 'start': 19, 'end': 20},
    #     {'entity': '北京景粼原著', 'type': 'PROJ', 'start': 25, 'end': 30},
    #     {'entity': '明天下午4点', 'type': 'TIME', 'start': 60, 'end': 65},
    #     {'entity': '/月', 'type': 'TIME', 'start': 78, 'end': 79}
    # ]
    # """

    # # bert4keras evaluate
    # print(cal_prf_ner())
