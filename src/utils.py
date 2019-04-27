#coding:utf-8
import os
import re
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import codecs
import json
import jieba
from joblib import dump, load
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

jieba.load_userdict('./models/nerDict.txt') #为结巴分词指定自定义词典


nerDict = None
dict_file = 'models/nerDict.txt'

def train_w2v_model(corpus, type='word', min_freq=2, size=100):
    """
    训练词向量
    :param corpus:
    :param type:
    :param min_freq:
    :param size:
    :return:
    """

    sentences = []

    for e in tqdm(corpus):
        sentences.append([i for i in e.strip().split() if i])
    print('训练集语料:', len(corpus))
    print('总长度: ', len(sentences))
    model = Word2Vec(sentences, size=size, window=5, min_count=min_freq)
    model.itos = {}
    model.stoi = {}
    model.embedding = {}
    item_to_id = {}
    print('保存模型...')

    for k in tqdm(sorted(list(model.wv.vocab.keys()), reverse=True)):
        item_to_id[k] = model.wv.vocab[k].index
    os.makedirs('./data/word2vec_models/', exist_ok=True)
    model.wv.save('./data/word2vec_models/word2vec.{}.{}d.model.kv'.format(type, size))
    pickle.dump(item_to_id, open('./data/{}_item_to_id.pkl'.format(type), 'wb'))
    words = model.wv.vocab
    with open('./data/word2vec_models/word2vec.{}.{}d.vocab.txt'.format(type, size, min_freq), 'w',
              encoding='UTF-8') as f:
        f.write('<S>\n</S>\n<UNK>\n')  # bilm-tf 要求vocab有这三个符号，并且在最前面
        for word in words:
            f.write(word + '\n')

def loadData(filePath):
    f = codecs.open(filePath,'r', 'utf-8')
    data = []
    for line in f.readlines():
        news = json.loads(line.strip())
        data.append(news)
    return data

def save_as_pd():

    train_data = loadData('../data/coreEntityEmotion_train.txt')
    test_data = loadData('../data/coreEntityEmotion_test_stage1.txt')

    columns = ["news_id", "title", "content", "coreEnt1", "coreEnt2", "coreEnt3", "Emo1", "Emo2", "Emo3"]
    train_df = pd.DataFrame(columns=columns)
    for idx, news in tqdm(enumerate(train_data)):
        cont_dict = {}
        cont_dict["news_id"] = news["newsId"]
        cont_dict["title"] = news["title"]
        cont_dict["content"] = news["content"]

        #分解coreEnt和Emo
        cols_c = ["coreEnt1", "coreEnt2", "coreEnt3"]
        cols_e = ["Emo1", "Emo2", "Emo3"]
        for i, col in enumerate(cols_c):
            if i < len(news["coreEntityEmotions"]):
                cont_dict[cols_c[i]] = news["coreEntityEmotions"][i]["entity"]
                cont_dict[cols_e[i]] = news["coreEntityEmotions"][i]["emotion"]
            else:
                cont_dict[cols_c[i]] = None
                cont_dict[cols_e[i]] = None

        #更新df
        train_df.loc[idx] = cont_dict

    columns_test = ["news_id", "title", "content"]
    test_df = pd.DataFrame(columns=columns_test)
    for idx, news in tqdm(enumerate(test_data)):
        cont_dict = {}
        cont_dict["news_id"] = news["newsId"]
        cont_dict["title"] = news["title"]
        cont_dict["content"] = news["content"]

        # 更新df
        test_df.loc[idx] = cont_dict

    ###分词
    train_df = cut(train_df)
    test_df = cut(test_df)
    os.makedirs('../data/csvs', exist_ok=True)
    dump(train_df, '../data/csvs/train_df.joblib')
    dump(test_df, '../data/csvs/test_df.joblib')
    # train_df.to_csv('../data/csvs/train_data.csv')
    # test_df.to_csv('../data/csvs/test_data.csv')

def loadNerDict(dict_file):
    """
    从文件中读取命名实体集
    :param dict_path:
    :return:
    """
    global nerDict
    nerDictFile = codecs.open(dict_file,'r','utf-8')
    nerDict = []
    for line in nerDictFile:
        nerDict.append(line.strip())
    nerDict = set(nerDict)

def getWords(news):
    '''
    get all word list from news
    :param news:
    :return:
    '''
    title = news['title']
    content = news['content']
    words = jieba.cut(title + '\t' + content)
    return list(words)

def getEntity(news):
    '''
    get all entity list from news
    :param news:
    :return:
    '''
    ners = []
    words = getWords(news)
    for word in words:
        if (word in nerDict):
            ners.append(word)
    return ners

def cut(cut_df):
    """
    按照字和词分词
    :param cut_df:
    :return:
    """
    chars = []
    words = []
    for s in tqdm(cut_df['content']):
        chars.append(' '.join(list(s)).strip())
        words.append(' '.join(jieba.cut(s, cut_all=False)).strip())
    cut_df['word'] = words
    cut_df['char'] = chars
    return cut_df

def main():
    save_as_pd()
    loadNerDict(dict_file)

if __name__ == "__main__":
    main()