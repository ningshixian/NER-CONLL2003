# -*- coding: utf-8 -*-
'''
针对 CONLL2003 实体识别任务语料处理及模型训练
'''
import numpy as np
from keras.models import Sequential, model_from_json, load_model
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import pickle
import codecs
import re
import gc
from keras.regularizers import l2


# 1 Initialize global variables
EMBEDDING_SIZE = 110 #(100 for word2vec embeddings and 11 for extra features (POS,CHUNK,CAP))
MAX_NB_WORDS = 200000
max_len=0
MAX_WORD_LENGTH=5
num_classes=5

batch_size = 128
epoch = 1

tags = ['None', 'Person', 'Location', 'Organisation', 'Misc']
train_file = 'data/train.txt'
test_file = 'data/test.txt'
val_file = 'data/valid.txt'
EMBEDDING_FILE = r'embedding/glove.100d.txt'
best_model_file = "model/best_model.h5"
prob_file = r'predict_result'
prf_file = 'prf_' + prob_file[:-15] + '.txt'


def pos(tag):
	onehot = np.zeros(5)
	if tag == 'NN' or tag == 'NNS':
		onehot[0] = 1
	elif tag == 'FW':
		onehot[1] = 1
	elif tag == 'NNP' or tag == 'NNPS':
		onehot[2] = 1
	elif 'VB' in tag:
		onehot[3] = 1
	else:
		onehot[4] = 1

	return onehot


def chunk(tag):
	onehot = np.zeros(5)
	if 'NP' in tag:
		onehot[0] = 1
	elif 'VP' in tag:
		onehot[1] = 1
	elif 'PP' in tag:
		onehot[2] = 1
	elif tag == 'O':
		onehot[3] = 1
	else:
		onehot[4] = 1

	return onehot


# 2 Build tokenized word index (建立字符索引字典)
def getDict(data_file):
	print('\n获取索引字典......')
	word_index = dict()
	indexVocab = []

	data = []
	for file in data_file:
		with codecs.open(file, encoding='utf-8') as train_f:
			for line in train_f:
				if line=='\r\n' or line=='\n': continue
				line = line.replace('\r\n', '')
				splited_line = line.split()
				data.append(splited_line[0])
	for word in data:
		if word in word_index:
			# A KeyError generally means the key doesn't exist.
			word_index[word] += 1
		else:
			word_index[word] = 1

	# 根据词频来确定每个词的索引
	wcounts = list(word_index.items())
	wcounts.sort(key=lambda x: x[1], reverse=True)
	sorted_voc = [wc[0] for wc in wcounts]
	word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
	print('Words in index: ', len(word_index))  # 18312

	# 加入 填充词 和 未登陆新词
	word_index['retain-padding'] = 0
	word_index['retain-unknown'] = len(word_index)
	return word_index


# 3 process GloVe embeddings
def read_vec(GLOVE_FILE, data_file):
	print('\n读词向量文件......')
	embeddings_index = dict()
	with codecs.open(GLOVE_FILE, encoding='utf-8') as f:
		for line in f:
			values = line.strip().split()
			if len(values) > 2:
				word = values[0]
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = embedding
	for file in data_file:
		with codecs.open(file, encoding='utf-8') as f:
			for line in f:
				if line in ['\r\n','\n']:
					continue
				w = line.split()[0]
				embed = embeddings_index.get(w)
				if embed is not None:
					if embed.shape==(110,):	# 可能一个词在不同的句子中有不同的词性，忽略
						continue
					embed = np.append(embed, pos(line.split()[1]))  # adding pos embeddings
					embed = np.append(embed, chunk(line.split()[2]))  # adding chunk embeddings
					# print(temp.shape)
					embeddings_index[w] = embed
				else:
					vec = np.random.uniform(-1, 1, size=EMBEDDING_SIZE)  # 随机初始化
					embeddings_index[w] = vec
	return embeddings_index


# 4 Prepare word embedding matrix
def get_embedding_matrix(word_index, EMBEDDING_FILE):
	print('\nPreparing embedding matrix.')
	data_file = [train_file, val_file, test_file]
	embeddings_index = read_vec(EMBEDDING_FILE, data_file)
	nb_words = min(MAX_NB_WORDS, len(word_index))
	print('\nnb_word: %d' % nb_words)
	pre_emb = np.zeros((nb_words, EMBEDDING_SIZE))
	for word, i in word_index.items():
		if i == 0: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			pre_emb[i] = embedding_vector
		else:
			vec = np.random.uniform(-1, 1, size=EMBEDDING_SIZE)  # 随机初始化
			pre_emb[i] = vec
	return pre_emb, nb_words


# 获取词的向量
def getEmb(pre_emb, w):
	randV = np.random.uniform(-0.25, 0.25, EMBEDDING_SIZE - 11)
	s = re.sub('[^0-9a-zA-Z]+', '', w)  # 字符串替换处理，然后返回被替换后的字符串
	arr = []
	if w == "~#~":
		arr = [0 for _ in range(EMBEDDING_SIZE)]
	elif w in pre_emb:
		arr = pre_emb[w]
	elif w.lower() in pre_emb:
		arr = pre_emb[w.lower()]
	elif s in pre_emb:
		arr = pre_emb[s]

	if len(arr) > 0:
		return np.asarray(arr)
	return randV


# 将一行转换为词索引序列
def sen_2_index(words, word_index, ctxWindows):
	x_sen = []
	# 首尾补零
	num = len(words)
	pad = int((ctxWindows - 1) / 2)
	for i in range(pad):
		words.insert(0, word_index['retain-padding'])
		words.append(word_index['retain-padding'])
	for i in range(num):
		x_sen.append(words[i:i + ctxWindows])
	return x_sen


# 5、将文本转化为词索引序列
def text_to_index_array(FILE_NAME, max_len):
	words = []
	tag = []
	sentence = []
	sentence_tag = []

	# get max words in sentence
	sentence_length = 0

	for line in open(FILE_NAME):
		if line in ['\n', '\r\n']:
			index_line = sen_2_index(words, word_index, 5)
			sentence.extend(index_line)
			sentence_tag.extend(tag)
			sentence_length = 0
			words = []
			tag = []
		else:
			assert (len(line.split()) == 4)
			if sentence_length > max_len:
				max_len = sentence_length
			sentence_length += 1

			word = line.split()[0]
			if word in word_index:
				words.append(word_index[word])  # 单词转索引
			else:
				# charVec.append(0)	# 索引字典里没有的词转为数字0
				words.append(word_index['retain-unknown'])

			t = line.split()[3]

			# Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
			if t.endswith('O'):  # none
				# tag.append(np.asarray([1, 0, 0, 0, 0]))
				tag.append(tags.index('None'))
			elif t.endswith('PER'):
				# tag.append(np.asarray([0, 1, 0, 0, 0]))
				tag.append(tags.index('Person'))
			elif t.endswith('LOC'):
				# tag.append(np.asarray([0, 0, 1, 0, 0]))
				tag.append(tags.index('Location'))
			elif t.endswith('ORG'):
				# tag.append(np.asarray([0, 0, 0, 1, 0]))
				tag.append(tags.index('Organisation'))
			elif t.endswith('MISC'):
				# tag.append(np.asarray([0, 0, 0, 0, 1]))
				tag.append(tags.index('Misc'))
			else:
				# tag.append(np.asarray([0, 0, 0, 0, 0]))
				print("error in input" + str(t))

	index_line = sen_2_index(words, word_index, 5)
	sentence.extend(index_line)
	sentence_tag.extend(tag)

	print("max sentence size is : " + str(max_len))
	assert (len(sentence) == len(sentence_tag))
	return np.asarray(sentence), sentence_tag


class computeF1(Callback):
    def __init__(self, x_test, y_test):
        # self.X_words_test = X_test
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        # X_test = self.X_words_test
        predictions = self.model.predict(self.x_test)
        y_pred = predictions.argmax(axis=-1)  # Predict classes
        self.y_test = self.y_test.argmax(axis=-1)
        print(len(y_pred), len(self.y_test))

        iprf_file = 'results/prf.txt'
        target = test_file

        p, r, f, c = predictLabels1(target, self.y_test, y_pred)

        with open(prf_file, 'a') as pf:
            print('write prf...... ')
            pf.write("epoch= " + str(epoch + 1) + '\n')
            pf.write("precision= " + str(pre) + '\t' + str(p) + '\n')
            pf.write("recall= " + str(rec) + '\t' + str(r) + '\n')
            pf.write("Fscore= " + str(f1) + '\t' + str(f) + '\n')
            pf.write("processed %d tokens with %d phrases;\n" % (c.token_counter, c.found_correct))
            pf.write('found: %d phrases; correct: %d.\n\n' % (c.found_guessed, c.correct_chunk))


def predictLabels1(target, y_test, y_pred, flag=None):
    s = []
    sentences = []
    n = 0
    with open(target) as f:
        for line in f:
            if line not in ['\n', '\r\n']:
                line = line.strip('\n').strip('\r\n').split(' ')
                s.append(line[0])
            else:
                for i in range(len(s)):
                	# if i >= maxlen_s: break
                    r = s[i] + '\t' + tags[y_test[n]] + '\t' + tags[y_pred[n]] + '\n'
                    print(r)
                    sentences.append(r)
                    n += 1
                sentences.append('\n')
                s = []
    with open('results/result.txt', 'w') as f:
        for line in sentences:
            f.write(str(line))

    p, r, f, c = conlleval.main((None, r'results/result.txt'))
    return round(Decimal(p), 2), round(Decimal(r), 2), round(Decimal(f), 2), c


if __name__ == '__main__':
	data_file = [train_file, val_file, test_file]
	word_index = getDict(data_file)
	nb_words = 30292
	
	# pre_emb, nb_words = get_embedding_matrix(word_index, EMBEDDING_FILE)
	
	# output = open('model/word_index.pkl', 'wb')
	# pickle.dump(word_index, output)  # 索引字典
	# output = open('model/pre_emb.pkl', 'wb')
	# pickle.dump(pre_emb, output)  # 词向量字典
	# word_index = pickle.load(open(u"model/word_index.pkl", "rb"))
	# pre_emb = pickle.load(open(u"model/pre_emb.pkl", 'rb'))
	# nb_words = len(pre_emb)

	x_train, y_train = text_to_index_array(train_file, max_len)
	x_test, y_test = text_to_index_array(test_file, max_len)
	x_val, y_val = text_to_index_array(val_file, max_len)

	# 转化为神经网络训练所用的张量
	x_train = pad_sequences(x_train, maxlen=5)
	y_train = to_categorical(np.asarray(y_train), num_classes)
	print('Shape of data  tensor:', x_train.shape)
	print('Shape of label tensor:', y_train.shape)

	x_test = pad_sequences(x_test, maxlen=5)
	y_test = to_categorical(np.asarray(y_test), num_classes)
	print('Shape of data  tensor:', x_test.shape)
	print('Shape of label tensor:', y_test.shape)

	x_val = pad_sequences(x_val, maxlen=5)
	y_val = to_categorical(np.asarray(y_val), num_classes)
	print('Shape of data  tensor:', x_val.shape)
	print('Shape of label tensor:', y_val.shape)

	model = Sequential()
	model.add(Embedding(input_dim=nb_words,  			# 索引字典大小
						output_dim=EMBEDDING_SIZE,  	# 词向量的维度
						# init='uniform',  				# 参数初始化方式是用均匀分布生成
						# weights=[pre_emb],  			# 初始化词向量表
						input_length=MAX_WORD_LENGTH,  	# 输入序列的长度
						# mask_zero=True  				# 对于较短的句子，用0来填充统一句子的长度，
						)) 								# mask_zero=True之后，对于0值就会忽略计算
	blstm = Bidirectional(CuDNNLSTM(units=100,
                                    return_sequences=False,
                                    kernel_regularizer=l2(1e-4),
                                    bias_regularizer=l2(1e-4)),
                            name='privateLSTM')
	model.add(blstm)
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	print(u'编译模型...')
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	model.summary()  # 打印出模型概况

	#############################################  Train Model  ############################################################

	# 该回调函数将在每个epoch后保存概率文件
	# write_prob = WritePRF(prob_file, prf_file, x_val, y_val)

	# 该回调函数将在每个迭代后保存的最好模型
	# check_call = checkpoint(best_model_file)

	computeF1 = computeF1(x_test, y_test)
	
	print(u'训练模型...')
	history = model.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epoch,
						callbacks=[computeF1],
						validation_data=(x_test, y_test))

	# model = load_model(best_model_file)

