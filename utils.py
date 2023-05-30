import re
import os
import string
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import codecs
import pickle as pkl
from decimal import Decimal
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from config import Config

config = Config()


def readBinEmbedFile(embFile, word_size):
    """
    读取二进制格式保存的词向量文件
    """
    import word2vec

    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    model = word2vec.load(embFile)
    print("加载词向量文件完成")
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()  # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


def readTxtEmbedFile(embFile, word_size):
    """
    读取预训练的词向量文件 
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with codecs.open(embFile, "r", "utf-8") as f:
        for line in tqdm(f):
            if len(line.split()) <= 2:
                continue
            values = line.strip().split()
            word = values[0].lower()
            vector = np.asarray(values[1:], dtype=np.float32)
            embeddings[word] = vector
        return embeddings


def readTxtEmbedFileForNER(embFile, word_size, word_index):
    """
    读取预训练的词向量文件 
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with codecs.open(embFile, "rb", "utf-8") as f:
        for line in tqdm(f):
            values = line.strip().split()
            if len(values) <= 2:
                continue
            word = values[0]
            if word in word_index:
                try:
                    vector = np.asarray(values[1:], dtype=np.float32)
                    embeddings[word] = vector
                except:
                    print(values)
        return embeddings


def readGensimFile(embFile):
    print("\nProcessing Embedding File...")
    import gensim

    model = gensim.models.Word2Vec.load(embFile)  # 'word2vec_words.model'
    word_vectors = model.wv
    return word_vectors


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split("-")[0] == "B":
            new_tags.append(tag)
        elif tag.split("-")[0] == "I":
            new_tags.append(tag)
        elif tag.split("-")[0] == "S":
            new_tags.append(tag.replace("S-", "B-"))
        elif tag.split("-")[0] == "E":
            new_tags.append(tag.replace("E-", "I-"))
        elif tag.split("-")[0] == "O":
            new_tags.append(tag)
        else:
            raise Exception("Invalid format!")
    return new_tags


def pad_word_chars(s, l, max_length):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    if len(s) >= max_length:
        s = s[:max_length]
        l = l[:max_length]
    else:
        padding = [0] * (max_length - len(s))
        s = s + padding
        l = l + padding
        # l = [list(np.eye(len(tag_to_id), dtype=int)[idx]) for idx in l]

    assert len(s) == len(l) == max_length
    return s, l


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def evaluate(target, y_pred, sentence_maxlen, idx2label):
    """
    Evaluate current model using CoNLL script.
    """
    s = []
    sentences = []
    s_num = 0
    with open(target) as f:
        for line in f:
            if not line == "\n":
                s.append(line.strip("\n"))
                continue
            else:
                prediction = y_pred[s_num]
                s_num += 1
                for i in range(len(s)):
                    if i >= sentence_maxlen:
                        break
                    r = s[i] + "\t" + idx2label[prediction[i]] + "\n"
                    sentences.append(r)
                sentences.append("\n")
                s = []
    # Write predictions to disk and run CoNLL script externally
    with open("../result/result.txt", "w") as f:
        for line in sentences:
            f.write(str(line))
    # CoNLL evaluation results
    p, r, f, c = conlleval.main((None, r"../result/result.txt"))
    return round(Decimal(p), 2), round(Decimal(r), 2), round(Decimal(f), 2), c


def load_data(filename, nb_sentence):
    """
    读取BIO的数据
    :param filename:BIO格式的语料
    :param nb_sentence:句子的数量
    :return:
    """
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


def get_labels():
    return ["O", "person", "time", "city", "project", "[CLS]", "[SEP]", ""]


class data_generator(DataGenerator):
    """数据迭代器
    定义 Dataset 类，封装一些数据读入和预处理方法
    """

    def __init__(self, label2id, train_data, tokenizer):
        super(data_generator, self).__init__(train_data, config.batch_size)
        self.label2id = label2id
        self.tokenizer = tokenizer  # 建立分词器

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < config.sequence_length:
                    token_ids += w_token_ids
                    if l == "O":
                        labels += [0] * len(w_token_ids)
                    else:
                        B = self.label2id[l] * 2 + 1
                        I = self.label2id[l] * 2 + 2
                        labels += [B] + [I] * (len(w_token_ids) - 1)
                else:
                    break
            token_ids += [self.tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                if config.pretrained_model_type:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                else:
                    batch_token_ids = sequence_padding(batch_token_ids, length=config.sequence_length)
                    batch_labels = sequence_padding(batch_labels, length=config.sequence_length)
                    yield batch_token_ids, batch_labels
                    batch_token_ids, batch_labels = [], []


def produce_matrix(token_to_id, embedFile):
    """词向量矩阵生成

    Parameters
    ----------
    word_index : [type]
        [description]
    embedFile : [type]
        [description]
    """
    if os.path.exists(config.embedding_pkl):
        print("从pkl加载词向量矩阵.......")
        with open(config.embedding_pkl, "rb") as f:
            emb_matrix = pkl.load(f)
        return emb_matrix

    word_embeddings = readTxtEmbedFileForNER(embedFile, 200, token_to_id)
    print("Found %s word vectors." % len(word_embeddings))  # 4706287

    miss_num = 0  # 未登陆词数量
    num = 0  # 登陆词数量
    num_words = len(token_to_id) + 1
    embedding_matrix = np.zeros((num_words, 200))
    for word, i in token_to_id.items():
        vec = None  # 初始化为空
        if word in word_embeddings:
            vec = word_embeddings.get(word)
            num = num + 1
        else:
            vec = word_embeddings["UNKNOWN_TOKEN"]  # 未登录词均统一表示
            miss_num = miss_num + 1
        embedding_matrix[i] = vec

    print("未登陆词数量", miss_num)  # 5
    print("登陆词数量", num)  # 5300

    # 保存数据文件(XX.pkl or XX_ngram.pkl)
    with open(config.embedding_pkl, "wb") as f:
        pkl.dump(embedding_matrix, f, -1)
    return embedding_matrix
