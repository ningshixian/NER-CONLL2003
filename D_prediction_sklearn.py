import joblib


def word2features(sent, i):
    """ 特征提取器 """
    word = sent[i]  # 词
    # postag = sent[i][1]   # 词性
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i + 1]

    # 特征可能不够.......
    features = {
        "bias": 1.0,
        "w": word,
        "w-1": prev_word,
        "w+1": next_word,
        "w-1:w": prev_word + word,
        "w:w+1": word + next_word,
        "word.isdigit()": word.isdigit(),
        # 'word.isalpha()':word.isalpha(),
        # 'postag':postag,
        # 'word.embedding': list(word_embeddings[word]) if word in word_embeddings else str(word_embeddings["UNKNOWN_TOKEN"])
    }
    return features


def sent2features(sent):
    """
    提取句子特征
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2tokens(sent):
    """
    提取句子词
    """
    return [token for token, postag, label in sent]


def get_entity(Y_pred, sents):
    """
    抽取句子中的所有实体
    """
    result = []
    entity = ""
    pre_tag, pre_class = "", ""

    for i in range(len(Y_pred)):
        entities = {"per": [], "xiaoqu": [], "location": [], "org": [], "time": []}
        pred = Y_pred[i]
        sent = sents[i]
        for j in range(len(pred)):
            tag = pred[j]
            w = sent[j]
            if tag.startswith("B") or tag.startswith("O"):
                if entity and pre_class:
                    entities[pre_class.lower()].append(entity)
                    entity = ""
                if tag.startswith("B"):
                    entity = w
                    pre_tag, pre_class = tag.split("_")[0], tag.split("_")[1]
                else:
                    pre_tag = "O"
                    pre_class = ""
            elif tag.startswith("M"):
                if pre_tag.startswith("B") or pre_tag.startswith("M"):
                    entity += w
                    pre_tag, pre_class = tag.split("_")[0], tag.split("_")[1]
                else:
                    pre_tag = "O"  # 错误标注改为'O'
                    pre_class = ""
            else:
                if pre_tag.startswith("B") or pre_tag.startswith("M"):
                    entity += w
                    pre_tag, pre_class = tag.split("_")[0], tag.split("_")[1]
                    entities[pre_class.lower()].append(entity)
                    entity = ""
                else:
                    pre_tag = "O"  # 错误标注改为'O'
                    pre_class = ""

        if entity and pre_class:
            entities[pre_class.lower()].append(entity)
            entity = ""
        result.append(entities)

    return result


if __name__ == "__main__":

    best_model_path = "../model/sklearn_crf.model"
    crf = joblib.load(best_model_path)
    sents = [
        r"宁时贤和周玉刚，2008年在北京的龙湖集团的数字科技部，住在天璞家园,老家是跳闸上海，职级是P4，手机号是13777878521，\
        想问问单据htcx20191234432处理得怎么样了",
        r"中共中央总书记、国家主席江泽民",
    ]
    X_train = [sent2features(s) for s in sents]
    Y_pred = crf.predict(X_train)
    # print(Y_pred)
    entities = get_entity(Y_pred, sents)
    print(entities)
