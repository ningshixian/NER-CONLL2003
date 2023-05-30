''' 
使用线性 CRF 实现实体识别的任务
使用 sklearn-crfsuite 中的 CRF 
识别实体类型：
    time: 时间
    location: 地点
    person_name: 人名
    org_name: 组织名
'''
import os
from itertools import chain  # 迭代器
import nltk
# nltk.download('conll2002')
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle as pkl
import codecs
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import joblib
from tqdm import tqdm


def get_data():
    idx2label = {0:'O', 1:'B_xiaoqu', 2:'M_xiaoqu', 3:'E_xiaoqu', 
                        4:'B_name', 5:'M_name', 6:'E_name', 
                        7:'B_time', 8:'M_time', 9:'E_time', 
                        10:'B_location', 11:'M_location', 12:'E_location', 
                        13:'B_org', 14:'M_org', 15:'E_org', 
                }
    train_sents = []
    train_sen = []

    print('Loading data...')
    data_files = ['data/longforNER/龙湖社区ner_conll.txt', 
                'data/ResumeNER/train.char.bmes', 
                'data/ResumeNER/dev.char.bmes',
                'data/ResumeNER/test.char.bmes',
                'data/boson/origindata_conll.txt',
                'data/MSRA/train1_conll.txt',
                'data/MSRA/testright1_conll.txt',
                'data/renMinRiBao/renmin_conll.txt',
                ]
    tag_not_seen = set()
    for data_file in data_files:
        with codecs.open(data_file, 'r', 'utf-8') as f:
            for line in tqdm(f):
                line = line.strip('\n').strip()
                if line:
                    splited = line.split('\t')
                    if splited[-1].lower()=='o': 
                        train_sen.append((splited[0], '?', 'O'))
                        continue
                    l = splited[1].split('_')[0].replace('S', 'B')
                    r = splited[1].split('_')[1].lower()
                    if r in ['time']:
                        train_sen.append((splited[0], '?', l+'_TIME'))
                    elif r in ['person_name', 'nr', 'name']:    # 人名
                        train_sen.append((splited[0], '?', l+'_PER'))
                    elif r in ['org_name', 'nt', 'org']:   # 组织名
                        train_sen.append((splited[0], '?', l+'_ORG'))
                    elif r in ['location','ns','loc']:   # 地名
                        train_sen.append((splited[0], '?', l+'_LOCATION'))
                    elif r in ['xiaoqu']:   # 小区名
                        train_sen.append((splited[0], '?', l+'_XIAOQU'))
                    else:
                        tag_not_seen.add(splited[1])
                else:
                    train_sents.append(train_sen)
                    train_sen = []
            if train_sen:
                train_sents.append(train_sen)
                train_sen = []

    print(tag_not_seen)
    print(len(train_sents))  # 63782
    import random
    train_sent_samples = random.sample(train_sents, 50)  #从list中随机获取5个元素，作为一个片断返回  
    # print(train_sent_samples)
    return train_sents


# ================================================================================= #


# print('获取词向量矩阵...')
# import utils
# embedFile = 'D:/wordEmbedding/Tencent_AILab_ChineseEmbedding_small.txt'
# word_index = []
# for sen in train_sents:
#     for w in sen:
#         word_index.append(w[0])
# word_index = list(set(word_index))
# word_embeddings = utils.readTxtEmbedFileForNER(embedFile, 200, word_index)  # 读取词向量
# print('Found %s word vectors.' % len(word_embeddings))  # 4706287


def word2features(sent,i):
    ''' 特征提取器 '''
    word = sent[i][0]     # 词
    postag = sent[i][1]   # 词性
    prev_word = "<s>" if i == 0 else sent[i-1][0]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1][0]
    
    # 特征可能不够.......
    features = {'bias':1.0,
                'w':word,
                'w-1': prev_word,
                'w+1': next_word,
                'w-1:w': prev_word+word,
                'w:w+1': word+next_word,
                'word.isdigit()':word.isdigit(),
                # 'word.isalpha()':word.isalpha(), 
                # 'postag':postag,
                # 'word.embedding': list(word_embeddings[word]) if word in word_embeddings else str(word_embeddings["UNKNOWN_TOKEN"]) 
                }
    return features
    
def sent2features(sent):
    '''
    提取句子特征
    '''
    return [word2features(sent,i) for i in range(len(sent))]
    
def sent2labels(sent):
    '''
    提取句子 label
    '''
    return [label for token,postag,label in sent]
    
def sent2tokens(sent):
    '''
    提取句子词
    '''
    return [token for token,postag,label in sent]



from collections import Counter
def print_transitions(trans_features):
    for (label_from,label_to),weight in trans_features:
        print("%-6s -> %-7s %.6f" %(label_from,label_to,weight))
 
def print_state_features(state_features):
    for (attr,label),weight in state_features:
        print("%.6f %-8s %s" %(weight,label,attr))


def load_model1(model_path):
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                            c1=0.1,
                            c2=0.1,
                            max_iterations=100,
                            all_possible_transitions=True)
    print('开始训练crf...')
    crf.verbose = 1
    crf.fit(X_train,Y_train)
    joblib.dump(crf, model_path)
    return crf


def load_model2(best_model_path):
    """随机搜索最佳模型"""
    if os.path.exists(best_model_path):
        print('best crf模型载入...')
        crf = joblib.load(best_model_path)
    else:
        # 定义超参数和参数查找空间
        crf = sklearn_crfsuite.CRF(
                algorithm = 'lbfgs',
                max_iterations = 100,
                all_possible_transitions = True)
        params_space = {'c1':scipy.stats.expon(scale = 0.5),
                        'c2':scipy.stats.expon(scale = 0.05)}
    
        # 使用相同的基准评估数据
        f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted',labels = labels)
        # 查询最佳模型
        rs = RandomizedSearchCV(estimator = crf, 
                                param_distributions = params_space,
                                cv = 3,
                                n_iter = 20,
                                verbose = 4,
                                # n_jobs = 1,  # -1 "timeout or by a memory leak.", UserWarning
                                scoring = f1_scorer)
        rs.fit(X_train,Y_train)
 
        # 输出最佳模型参数
        print("The Best Params:",rs.best_params_)   # The Best Params: {'c1': 0.00255, 'c2': 0.00742}
        print("The Best CV score:",rs.best_score_)  # 0.9721625779172182
        print("Model Size:{:.2f}M".format(rs.best_estimator_.size_ / 1000000))  # 5.79M
 
        crf = rs.best_estimator_
        joblib.dump(crf, best_model_path)
    return crf


if __name__ == "__main__":

    if os.path.exists('data/data_sklearn.pkl'):
        print('提取数据.pkl')
        with codecs.open('data/data_sklearn.pkl', 'rb') as f:
            (X_train, Y_train, X_test, Y_test) = pkl.load(f)
        print('提取数据finish!')
    else:
        train_sents = get_data()
        # 划分训练/测试集
        train_sents, test_sents = train_test_split(train_sents, test_size=0.2)
        print('特征抽取....')
        X_train = [sent2features(s) for s in train_sents]
        Y_train = [sent2labels(s) for s in train_sents]
        X_test = [sent2features(s) for s in test_sents]
        Y_test = [sent2labels(s) for s in test_sents]
        print('保存数据.pkl')
        with codecs.open('data/data_sklearn.pkl', 'wb') as f:
            pkl.dump((X_train, Y_train, X_test, Y_test), f, -1)

    model_path = 'model/sklearn_crf.model'    
    crf = load_model1(model_path)
    
    # 获得标记是 B 或者 I 的结果
    labels = list(crf.classes_)
    print(labels) 
    labels.remove('O')

    print('使用测试集评测 & Evaluate...')
    Y_pred = crf.predict(X_test)
    # print(Y_pred[:2])
    metrics.flat_f1_score(Y_test,Y_pred,average='weighted',labels = labels)

    sorted_labels = sorted(labels, key = lambda x:(x[1:],x[0]))
    print("初始模型效果如下...")
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    print(classification_report(Y_test, Y_pred))
    print(metrics.flat_classification_report(Y_test,Y_pred,
                                            labels = sorted_labels,
                                            digits = 3)) # digits 表示保留几位小数

    # print(X_test[:5])
    # print(Y_test[:5])
    # print(Y_pred[:5])

    # ================================================================================= #

    best_model_path = 'model/sklearn_crf.bestmodel'    
    crf = load_model2(best_model_path)

    labels = list(crf.classes_)
    labels.remove('O')
    sorted_labels = sorted(labels, key = lambda x:(x[1:],x[0]))
    
    Y_pred = crf.predict(X_test)
    print("最佳模型效果如下...")
    print(metrics.flat_classification_report(Y_test,Y_pred,
                                            labels = sorted_labels,
                                            digits = 3))
 
    print("\n最大转移概率")
    print_transitions(Counter(crf.transition_features_).most_common(20))
 
    print("\n最低转移概率")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
 
    print("\nTop Positive")
    print_state_features(Counter(crf.state_features_).most_common(30))
 
    print("\nTop Negative")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])


