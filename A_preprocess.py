import io, os
import codecs
from tqdm import tqdm
import ahocorasick
import paddlehub as hub
import sys
import numpy as np
from random import shuffle

sys.path.append(os.getcwd())  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config

config = Config()
np.random.seed(1671)  # 随机数种子，重复性设置

"""NER语料处理和标注
暂时只用到人民日报、MSRA和longfor这3个语料，有待扩充

包含实体类别：人名 person 时间 time 城市 city 项目名 project (仅longfor语料)
"""


def load_project():
    """加载项目词典
    
    Returns
    -------
    AC自动机
        项目词典 树形结构 便于快速查找
    """
    projects = []
    pro_path1 = "data/dict/longfor_project_new.txt"
    pro_path2 = "data/dict/longfor_project_nocity.txt"
    with io.open(pro_path1, "r", encoding="utf8") as f:
        for line in f:
            projects.append(line.strip("\n"))
            projects.append(line.strip("\n").replace("项目", ""))
    with io.open(pro_path2, "r", encoding="utf8") as f:
        for line in f:
            projects.append(line.strip("\n").split("\t")[0])
    # AC-自动机
    A = ahocorasick.Automaton()
    for index, word in enumerate(projects):
        A.add_word(word, (index, word))
    A.make_automaton()
    return A


def load_city():
    """加载城市词典
    
    Returns
    -------
    AC自动机
        城市词典 树形结构 便于快速查找
    """
    china_city = set()
    city_path = "data/dict/china_city.txt"
    with io.open(city_path, "r", encoding="utf8") as f:
        for line in f:
            cols = line.split("\t")
            china_city.add(cols[2])
    # AC-自动机
    B = ahocorasick.Automaton()
    for index, word in enumerate(list(china_city)):
        B.add_word(word, (index, word))
    B.make_automaton()
    return B


city_ac = load_city()
project_ac = load_project()
lac = hub.Module(name="lac")


def iob_ranges(words, tags):
    """从标注预料中抽取实体 IOB -> entities
    
    Parameters
    ----------
    来自 {'word': ['今天', '天气', '很好'], 'tag': ['TIME', 'n', 'a']}
    words : list
        类似 ['今天', '天气', '很好']
    tags : list
        类似 ['TIME', 'n', 'a']
    
    Returns
    -------
    list
        ranges: 保存实体及其对应类型
    """
    assert len(words) == len(tags)
    ranges = []
    entity, entity2, entity3 = "", "", ""

    for i, tag in enumerate(tags):
        # 时间
        if tag == "t" or tag == "TIME":
            entity += words[i]
            e_type1 = "time"
            if i == len(tags) - 1 or (tags[i + 1] != "t" and tags[i + 1] != "TIME"):
                ranges.append((entity, e_type1))
                entity = ""
        # 人名
        elif tag == "nr" or tag == "PER":
            entity3 += words[i]
            e_type3 = "person"
            if i == len(tags) - 1 or (tags[i + 1] != "nr" and tags[i + 1] != "PER"):
                ranges.append((entity3, e_type3))
                entity3 = ""
    # 城市（可能会漏掉城市简写）
    ents = []
    for item in city_ac.iter("".join(words)):  # (3, (1, '英国'))
        e_type2 = "city"
        ents.append((item[1][1], e_type2))
    ranges.extend(ents)

    ranges = list(set(ranges))
    ranges = sorted(ranges, key=lambda x: len(x[0]), reverse=False)  # 避免 ‘10.29’ ‘2017.10.29’ 冲突
    return ranges


def run_lacParse(text):
    """使用 Paddlehub LAC 服务
    """
    inputs = {"text": [text]}
    results = lac.lexical_analysis(data=inputs)  # IndexError: string index out of range 未解决
    results = results[0]  # 仅需取第一句
    return results["word"], results["tag"]


def tagging(sent, ents):
    """改造数据，加入实体标注{{}}
    1、对ents逆序排序
    2、每匹配到sent中的ent，替换为{{number}}，并保存到字典{number:ent}
    3、最后再统一替换
    
    Parameters
    ----------
    sent : str
        '北京公司建设的北京天街项目'
    ents : list
        [('北京', 'city'), ('北京天街', 'project')]
    
    Returns
    -------
    [str]
        '{{city:北京}}公司建设的{{project:北京天街}}项目'
    """
    match_dict = {}
    ents = sorted(list(set(ents)), key=lambda x: len(x[0]), reverse=True)
    # print(ents)
    for i in range(len(ents)):
        e = ents[i][0]
        e_type = ents[i][1]
        sent = sent.replace(e, "{{" + str(i) + "}}")
        match_dict["{{" + str(i) + "}}"] = "{{" + e_type + ":" + e + "}}"
    for key, value in match_dict.items():
        sent = sent.replace(key, value)
    return sent


def is_number(s):
    try:
        # 因为使用float有一个例外是'NaN'
        if s == "NaN":
            return False
        float(s)
        return True
    except ValueError:
        return False


def cutShortItem(l):
    """
    将列表中 被其他元素包含的较短元素 删掉
    比如：['天街', '北京天街', '原著'] → ['北京天街', '原著']
    """
    l = list(set(l))
    l = sorted(l, key=lambda x: len(x), reverse=False)
    new_l = []
    for i in range(len(l)):
        if l[i] not in "".join(l[:i] + l[i + 1 :]):
            new_l.append(l[i])
    return new_l


def convertMSRA(path1, path2, mode="train"):
    """将原始数据转格式换为XXX{{YYY}}XXX格式
    
    原始数据格式
    train: 北京图书馆/ns
    test: 北/B_ns 京/M_ns 图/M_ns 书/M_ns 馆/E_ns 等/o
    
    原始数据的标注类型
    人名nr、地名ns、机构名nt （只要nr）    
    
    Parameters
    ----------
    path1 : str
        数据的存放路径
    path2 : str
        处理后存放路径 final_path
    mode : str, optional
        区分训练集和测试集的处理方法
    """
    output_data = codecs.open(path2, "a", "utf-8")
    with codecs.open(path1, "r", "utf-8") as f:
        lines = f.readlines()

    for i in tqdm(range(len(lines[:pages]))):
        line = lines[i].replace("{", "").replace("}", "").strip().split()
        if len(line) == 0:
            continue
        ents = []  # 实体记录
        new_line = []  # 加入{{}}标记后的句子
        for word in line:
            splited = word.split("/")
            if mode == "train":
                if splited[1] == "nr":
                    ents.append((splited[0], "person"))
                new_line.append(splited[0])  # 非实体词
            elif mode == "test":
                if splited[1] == "B_nr":  # 存在连续两个 B_nr B_nr 的情况
                    ents.append((splited[0], "person"))
                elif splited[1] == "M_nr":
                    ents[-1] = (ents[-1][0] + splited[0], ents[-1][1])
                elif splited[1] == "E_nr":
                    ents[-1] = (ents[-1][0] + splited[0], ents[-1][1])
                new_line.append(splited[0])  # 非实体词

        new_line = "".join(new_line)
        words, labels = run_lacParse(new_line)
        ents_2 = iob_ranges(words, labels)
        for item in ents_2:
            if item[1] == "person":
                continue
            ents.append(item)
        new_line = tagging(new_line, ents)

        if "{{" not in new_line:
            continue  # 无实体的句子剔除

        assert new_line.count("{") == new_line.count("}")
        output_data.write(new_line)
        output_data.write("\n")
    output_data.close()


def convertRenMinRiBao(path1, path2):
    """将原始数据转格式换为XXX{{YYY}}XXX格式
    
    原始数据格式
    当/o 希望工程/o 救助/o 的/o 百万/o ... 
    
    原始数据的标注类型
    时间词t、数词m、人名nr、地名ns、机构名nt、...（只要t nr）    
    
    Parameters
    ----------
    path1 : str
        数据的存放路径
    path2 : str
        处理后存放路径 final_path
    """
    output_data = codecs.open(path2, "a", "utf-8")
    with codecs.open(path1, "r", "utf-8") as f:
        lines = f.readlines()

    for i in tqdm(range(len(lines[:pages]))):
        line = lines[i].replace("{", "").replace("}", "").strip()
        line = line.split()[1:]
        while "" in line:
            line.remove("")
        if len(line) == 0:
            continue
        words = []
        labels = []
        for word in line:
            word = word.split("[")[-1].split("]")[0]
            splited = word.split("/")
            words.append(splited[0])
            labels.append(splited[1])

        ents = []  # 实体记录
        new_line = "".join(words)  # 加入{{}}标记后的句子
        ents = iob_ranges(words, labels)
        new_line = tagging(new_line, ents)

        if "{{" not in new_line:
            continue  # 无实体的句子剔除

        assert new_line.count("{") == new_line.count("}")
        output_data.write(new_line)
        output_data.write("\n")
    output_data.close()


# 未使用
def convertResume(path1, path2):
    # path1 = 'data/ResumeNER/'
    for file_name in os.listdir(path1):
        sents = []
        with codecs.open(path1 + file_name, "r", "utf-8") as f:
            for line in f:
                line = line.replace(" ", "\t").replace("-", "_")
                sents.append(line)
        with codecs.open(path1 + file_name, "w", "utf-8") as f:
            for line in sents:
                f.write(line)


def convertLongfor(path1, path2):
    """将原始数据转格式换为XXX{{YYY}}XXX格式
    
    原始数据格式
    id	title	start_time	end_time	content	hits	...
    
    Parameters
    ----------
    path1 : str
        数据的存放路径 longforNER/kno_faq_clean.txt
    path2 : str
        处理后存放路径 final_path
    """
    output_data = codecs.open(path2, "a", "utf-8")
    with codecs.open(path1, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entity_filter_list = "龙信|龙建|龙小湖|龙湖|龙客|拉风|小冠|小寓|平米|奠仪|商保|尔东|木子李|弓长张|咨询师|心理咨询师|奠仪|商保|仕官生|士官生|安家费|保理|"
    entity_filter_list = entity_filter_list.split("|")

    cnt = 0  # 格式错误句子数
    for i in tqdm(range(1, len(lines[:pages]))):
        row = lines[i].replace("{", "").replace("}", "").strip()
        row = row.split("\t")
        if len(row) == 0:
            continue
        # 部分格式错误句子处理
        if not len(row) > 4:
            cnt += 1
            continue
        for line in [row[1], row[4]]:
            line = line.strip()
            if len(line) == 0:
                continue
            # 项目实体标注
            ents = []
            new_line = line[:]  # 加入{{}}标记
            for item in project_ac.iter(line):  # (3, (1, 'XX原著'))
                e_type2 = "project"
                ents.append((item[1][1], e_type2))
            # for e in cutShortItem(ents):
            #     new_line = new_line.replace(e, '{{project:' + e + '}}')
            #     # lac_line = lac_line.replace(e, 'P'*len(e))

            words, labels = run_lacParse(line)
            entities = iob_ranges(words, labels)
            for item in entities:
                # 过滤LAC抽取出的数字时间
                if item[0] in entity_filter_list or is_number(item[0]):
                    continue
                ents.append(item)
            new_line = tagging(new_line, ents)

            if "{{" not in new_line:
                continue  # 无实体的句子剔除

            assert new_line.count("{") == new_line.count("}")
            output_data.write(new_line)
            output_data.write("\n")
    output_data.close()


def convert2conll(path1, path2):
    """将经过{{}}标注的数据，转换成CONLL格式
    
    Parameters
    ----------
    path1 : 经过{{}}标注的数据存放路径
    path2 : CONLL格式的数据路径
    """
    print("\n将原始数据转换为CONLL格式......")
    input_data = codecs.open(path1, "r", "utf-8")
    output_data = codecs.open(path2, "w", "utf-8")
    for line in input_data.readlines():
        i = 0
        line = line.strip()
        if "{{" not in line:
            continue  # 无实体的句子剔除
        try:
            while i < len(line):
                if line[i] == "{":
                    i += 2
                    temp = ""
                    while line[i] != "}":
                        temp += line[i].strip()
                        i += 1
                    i += 2
                    splited = temp.split(":", 1)  # num -- 分割次数。默认为 -1, 即分隔所有。
                    ent_name = splited[1]
                    for j in range(len(ent_name)):
                        w = ent_name[j]
                        # BIO标注格式
                        if j == 0:
                            output_data.write(w + "\t" + "B_" + splited[0] + "\n")
                        else:
                            output_data.write(w + "\t" + "I_" + splited[0] + "\n")
                        # # # BMEO标注格式
                        # if j == 0:
                        #     output_data.write(w + "\t" + "B_" + splited[0] + "\n")
                        # elif j == (len(ent_name) - 1):
                        #     output_data.write(w + "\t" + "E_" + splited[0] + "\n")
                        # else:
                        #     output_data.write(w + "\t" + "M_" + splited[0] + "\n")
                else:
                    output_data.write(line[i] + "\t" + "O" + "\n")
                    i += 1
            output_data.write("\n")
        except Exception as e:
            print(str(e))
            print(line)
    input_data.close()
    output_data.close()


def cut_train_dev_test(conll_path):
    """切分数据集
        →训练集、验证集、测试集
    """
    print("\n切分数据集......")

    lines, temp = [], []
    with codecs.open(conll_path, "r", "utf-8") as f:
        for line in tqdm(f):
            if not line == "\n":
                temp.append(line)
            else:
                lines.append(temp)
                temp = []

    # 打乱数据集
    shuffle(lines)

    print("\n写入文件......")
    a = int(len(lines) * 0.9)
    b = int(len(lines) * 0.95)
    with codecs.open(config.train_data, "w", "utf-8") as f:
        for line in tqdm(lines[:a]):
            for xx in line:
                f.write(xx)
            f.write("\n")

    with codecs.open(config.dev_data, "w", "utf-8") as f:
        for line in tqdm(lines[a:b]):
            for xx in line:
                f.write(xx)
            f.write("\n")

    with codecs.open(config.test_data, "w", "utf-8") as f:
        for line in tqdm(lines[b:]):
            for xx in line:
                f.write(xx)
            f.write("\n")


def main():
    len_treshold = config.sequence_length - 2  #  每条数据的最大长度, 留下两个位置给[CLS]和[SEP]
    final_path = config.processed_data_dir
    conll_path = config.final_conll_data_dir
    msra_train = config.source_data_MSRA_train
    msra_test = config.source_data_MSRA_test
    rmrb_data = config.source_data_renMinRiBao
    long_data = config.source_data_longfor

    # 微软NER数据-预处理
    convertMSRA(msra_train, final_path, mode="train")
    convertMSRA(msra_test, final_path, mode="test")

    # 人命日报NER数据-预处理
    convertRenMinRiBao(rmrb_data, final_path)

    # 龙湖内部数据-预处理
    convertLongfor(long_data, final_path)

    # 转换CONLL格式
    convert2conll(final_path, conll_path)

    # 切分数据集
    cut_train_dev_test(conll_path)


if __name__ == "__main__":
    pages = 50000000000  # 50 50000000000
    main()
