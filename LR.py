# coding:utf-8

import os
import time
import jieba
import codecs
import json
from model import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from joblib import dump, load
import re
from scipy.sparse import vstack
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from src.utils import train_w2v_model

jieba.load_userdict('./models/nerDict.txt') #为结巴分词指定自定义词典

class LR_ent(BaseModel):
    def __init__(self):
        # self.train_df = load('../data/csvs/train_df.joblib')
        # self.test_df = load('../data/csvs/test_df.joblib')
        self.train_dt = None
        self.test_dt = None

    def preprocessing(self):
        """
        1、读入实体词典
        2、读入预料库，生成TFidf矩阵
        3、将原始训练集划分为训练集和验证集
        :return:
        """

        self.train_dt = self.loadData('./data/coreEntityEmotion_train.txt')
        self.test_dt = self.loadData('./data/coreEntityEmotion_test_stage1.txt')

        # load nerDict as named entity recognizer
        self.loadNerDict()

        # train tfIdf as core entity score model

        print("loading all ner corpus from train data...")

        if not os.path.exists('models/nerCorpus.joblib'):
            # self.news_vocab_lst = []
            nerCorpus = []
            all_data = self.train_dt + self.test_dt
            for news in tqdm(all_data):
                t = self.getEntity(news)
                # self.news_vocab_lst.append(t)
                nerCorpus.append(' '.join(t))
            dump(nerCorpus, 'models/nerCorpus.joblib')
            # dump(self.news_vocab_lst, 'models/news_vocab_lst.joblib')
            # print(nerCorpus)
        else:
            nerCorpus = load('models/nerCorpus.joblib')
            # self.news_vocab_lst =load('models/news_vocab_lst.joblib')

        ###将语料库转化为word2vec模型
        if not os.path.exists('./data/word2vec_models/word2vec.word.100d.model.kv'):
            train_w2v_model(nerCorpus)
        self.wv = KeyedVectors.load('./data/word2vec_models/word2vec.word.100d.model.kv', mmap='r')

        print(self.wv["机器视觉"])

        ###将原始样本集划分为用于模型选择的训练集和验证集
        self.train_dt_ms, self.eval_dt_ms = self.split_train_eval_data(self.train_dt)

    def fit(self, type="train"):
        '''
        train model for coreEntity
        Baseline use entityDict for named entity recognition, you can use a more wise method.
        Baseline use tfIdf score as feature and LR as classification model
        :return:
        '''

        if type == "MS":
            train_data = self.train_dt_ms
        else:
            train_data = self.train_dt

        vocabFile = codecs.open('./data/word2vec_models/word2vec.word.100d.vocab.txt','r','utf-8')
        self.vocab = []
        for line in vocabFile:
            self.vocab.append(line.strip())
        self.vocab = set(self.vocab)

        if not os.path.exists('models/CoreEntityCLF.joblib'):
            # 2. train LR with tfIdf score as features
            isCoreX = []
            isCoreY = []

            for news in tqdm(train_data):

                # title_lst = list(jieba.cut(news["title"], cut_all=False))

                content_lst = self.getEntity(news)

                coreEntity_GroundTruth_t = [x['entity'] for x in news['coreEntityEmotions']]
                coreEntity_GroundTruth = []
                for s in coreEntity_GroundTruth_t:
                    coreEntity_GroundTruth.extend(list(jieba.cut(s, cut_all=False)))

                for idx, word in enumerate(content_lst):
                    if word not in self.vocab:
                        continue
                    label = 0
                    if word in coreEntity_GroundTruth:
                        label = 1
                    isCoreX.append(self.wv[word])
                    isCoreY.append(label)

            # 3. train LR model for coreEntity

            print("training LR model for coreEntity...")
            self.coreEntityCLF = LogisticRegression(random_state=0, solver='lbfgs',
                                     multi_class='multinomial').fit(isCoreX, isCoreY)
            dump(self.coreEntityCLF, 'models/CoreEntityCLF.joblib')
        else:
            print("loading LR model for file...")
            self.coreEntityCLF = load('models/CoreEntityCLF.joblib')

    def predict(self, type="test"):
        """
        预测输出
        :return:
        """
        # print("length of test data: %d" %len(testData))

        # print(self.coreEntityCLF.classes_)

        if type == "MS":
            test_data = self.eval_dt_ms
        else:
            test_data = self.test_dt

        self.en_lst = []
        self.em_lst = []
        for news in tqdm(test_data):
            # print(news)

            # s_time = time.clock()

            # title_lst = list(jieba.cut(news["title"], cut_all=False))

            cont_lst = self.getEntity(news)

            # predict core Entities
            coreEntities = []
            count = 0
            for idx, word in enumerate(cont_lst):
                if word not in self.vocab:
                    continue
                sample = self.wv[word]
                proba = self.coreEntityCLF.predict_proba([sample])[0]
                # print(name, proba)
                rescale_proba = proba[1] / proba[0] * 300
                if (rescale_proba > 1):
                    coreEntities.append(word)
                    count += 1
                    # print(name, rescale_proba)

                if count == 3:
                    break

            # e_time = time.clock()
            # print("1# time consuming: %f" %(e_time - s_time))

            # s_time = time.clock()
            # predict emotion of core entity
            entEmotions = []
            for entity in coreEntities:
                text = news['title'] + '\n' + news['content']
                relatedSents = []
                for sent in re.split(r'[\n\t，。！？“”（）]', text):
                    if (entity in sent):
                        relatedSents.append(sent)
                relatedText = ' '.join(relatedSents)
                emotionTfIdfFeature = self.emotionTfIdf.transform([relatedText]).toarray()
                emotion = self.emotionCLF.predict(emotionTfIdfFeature)
                entEmotions.append(emotion[0])

                # entEmotions.append("POS")

            # e_time = time.clock()
            # print("#2 time consuming: %f" % (e_time - s_time))

            self.en_lst.append(coreEntities)
            self.em_lst.append(entEmotions)

        if type == "test":
            self.output()

    def output(self, type="test"):
        """
        输出结果
        :return:
        """
        f_submit = codecs.open('data/submission_4.txt',
                               'w', 'utf-8')

        if type == "MS":
            test_data = self.eval_dt_ms
        else:
            test_data = self.test_dt

        for news, ents, emos in zip(test_data, self.en_lst, self.em_lst):
            f_submit.write(news['newsId'])
            f_submit.write('\t')
            f_submit.write(','.join(ents))
            f_submit.write('\t')
            f_submit.write(','.join(emos))
            f_submit.write('\n')

    def predict_score(self, type="MS"):
        """
        用于模型验证，
        :return: 预测结果的评分
        """

        self.predict(type="MS")

        F1_ent = []
        F1_emo = []

        for news, ent_predict, emo_predict in tqdm(zip(self.eval_dt_ms, self.en_lst, self.em_lst)):
            coreEnt = [x['entity'] for x in news['coreEntityEmotions']]
            coreEmo = [x['emotion'] for x in news['coreEntityEmotions']]

            ###计算实体F1值
            count_ent = 0
            for ent in ent_predict:
                if ent in coreEnt:
                    count_ent += 1
            precision_ent = count_ent / len(ent_predict)
            recall_ent = count_ent / len(coreEnt)
            if precision_ent != 0 and recall_ent != 0:
                F1_ent.append(2 * (precision_ent * recall_ent) / (precision_ent + recall_ent))
            else:
                F1_ent.append(0)

            ###计算情感F1值
            coreEmo_str = ["%s_%s" %(ent, emo) for (ent, emo) in zip(coreEnt, coreEmo)]
            em_lst_str = ["%s_%s" %(ent, emo) for (ent, emo) in zip(ent_predict, emo_predict)]
            count_emo = 0
            for em in em_lst_str:
                if em in coreEmo_str:
                    count_emo += 1
            precision_em = count_emo / len(em_lst_str)
            recall_em = count_emo / len(coreEmo_str)
            if precision_em != 0 and recall_em != 0:
                F1_emo.append(2 * (precision_em * recall_em) / (precision_em + recall_em))
            else:
                F1_emo.append(0)

        ###计算综合得分
        F1_ent_score = np.array(F1_ent).mean()
        F1_emo_score = np.array(F1_emo).mean()
        F1_score = (F1_emo_score + F1_ent_score) / 2

        # print(F1_ent, F1_emo)

        print("final evaluation score is (%f, %f, %f)" %(F1_score, F1_ent_score, F1_emo_score))

        return F1_score

    def trainEmotion(self, type="test"):
        '''
        train emotion model
        Baseline use tfIdf vector as feature, linearSVC as classfication model
        :return:
        '''

        if type == "MS":
            train_data = self.train_dt_ms
        else:
            train_data = self.train_dt

        if not os.path.exists('models/emotionX.joblib'):

            emotionX = []
            emotionY = []

            print("loading emotion corpus from train data...")

            # 1. get all related sentences to the entities
            for news in tqdm(train_data):

                text = news['title'] + '\n' + news['content']
                entities = [x['entity'] for x in news['coreEntityEmotions']]
                emotions = [x['emotion'] for x in news['coreEntityEmotions']]
                entityEmotionMap = dict(zip(entities, emotions))
                entitySentsMap = {}
                for entity in entityEmotionMap.keys():
                    entitySentsMap[entity] = []

                ###寻找每一个命名实体对应的句子，值为列表
                for sent in re.split(r'[\n\t，。！？“”（）]', text):
                    for entity in entityEmotionMap.keys():
                        if (entity in sent):
                            entitySentsMap[entity].append(sent)

                for entity, sents in entitySentsMap.items():
                    relatedText = ' '.join(sents)
                    emotionX.append([relatedText]) #值为将包含命名实体的句子用空格连接的字符串，用列表括起来
                    emotionY.append(entityEmotionMap[entity]) #值为情感

            dump(emotionX, 'models/emotionX.joblib')
            dump(emotionY, 'models/emotionY.joblib')
        else:
            print("loading emotionX and emotionY from file...")
            emotionX = load('models/emotionX.joblib')
            emotionY = load('models/emotionY.joblib')

        # 2. train tf-idf model for emotion related words
        if not os.path.exists('models/emotionTfIdf.joblib'):
            print("fitting emotion tfIdf model...")

            emotionWordCorpus = []
            all_data = self.train_dt + self.test_dt
            for news in tqdm(all_data):
                emotionWordCorpus.append(' '.join(self.getWords(news)))
            self.emotionTfIdf = TfidfVectorizer()
            self.emotionTfIdf.fit(emotionWordCorpus)
            dump(self.emotionTfIdf, 'models/emotionTfIdf.joblib')
        else:
            print("loading emotion tfIdf model...")
            self.emotionTfIdf = load('models/emotionTfIdf.joblib')

        if not os.path.exists('models/emotionCLF.joblib'):

            # 3. use naive bayes to train emotion classifiction
            s_time = time.clock()
            lst = [self.emotionTfIdf.transform(x) for x in emotionX]
            e_time = time.clock()
            print("time consuming: %f" %(e_time - s_time))

            emotionX = vstack(lst).tocsr()

            print("training emotion clf with linearSVC...")

            print(emotionX.shape)
            self.emotionCLF = MultinomialNB()
            self.emotionCLF.fit(emotionX, emotionY)

            print(self.emotionCLF.score(emotionX, emotionY))

            dump(self.emotionCLF, 'models/emotionCLF.joblib')
        else:
            self.emotionCLF = load('models/emotionCLF.joblib')

    def getTfIdfScore(self, news, tfIdf):

        doc = self.getEntity(news)
        tfIdfFeatures = tfIdf.transform([' '.join(doc)])

        tfIdfScores = tfIdfFeatures.data
        # normalize
        tfIdfScoresNorm = normalize([tfIdfScores], norm='max')

        tfIdfNameScore = [(self.featureName[x[0]], x[1]) for x in zip(tfIdfFeatures.indices, tfIdfScoresNorm[0])]
        tfIdfNameScore = sorted(tfIdfNameScore, key=lambda x: x[1], reverse=True)

        return tfIdfNameScore

    def loadNerDict(self):
        nerDictFile = codecs.open('models/nerDict.txt','r','utf-8')
        self.nerDict = []
        for line in nerDictFile:
            self.nerDict.append(line.strip())
        self.nerDict = set(self.nerDict)

    def getWords(self, news):
        '''
        get all word list from news
        :param news:
        :return:
        '''
        title = news['title']
        content = news['content']

        words = jieba.cut(title + '\t' + content)

        return list(words)

    def getEntity(self, news):
        '''
        get all entity list from news
        :param news:
        :return:
        '''
        ners = []
        words = self.getWords(news)
        for word in words:
            if (word in self.nerDict):
                ners.append(word)
        return ners
        # return words

    def loadData(self, filePath):
        f = codecs.open(filePath,'r', 'utf-8')
        data = []
        for line in f.readlines():
            news = json.loads(line.strip())
            data.append(news)
        return data

    def testCoreEntity(self):

        self.coreEntityTfIdf = load('models/nerTfIdf.joblib')
        self.coreEntityCLF = load('models/coreEntityCLF.joblib')
        self.featureName = self.coreEntityTfIdf.get_feature_names()

        # testData = self.loadData('data/coreEntityEmotion_train.txt')
        testData = self.loadData('data/coreEntityEmotion_test_stage1.txt')

        f_submit = codecs.open('data/submission_4.txt',
                                        'w', 'utf-8')

        # print("length of test data: %d" %len(testData))

        # print(self.coreEntityCLF.classes_)

        for news in tqdm(testData):
            # print(news)
            predictCoreEntityEmotion = {}

            tfIdfNameScore = self.getTfIdfScore(news, self.coreEntityTfIdf)

            # s_time = time.clock()

            title_lst = list(jieba.cut(news["title"], cut_all=False))

            # predict core Entities
            coreEntities = []
            count = 0
            len_news = len(tfIdfNameScore)
            for ind, (name, score) in enumerate(tfIdfNameScore):
                sample = [score, 0, 0]
                if ind < 0.1 * len_news:
                    sample[2] = 1
                if name in title_lst:
                    sample[1] = 1
                proba = self.coreEntityCLF.predict_proba([sample])[0]
                # print(name, proba)
                rescale_proba = proba[1] / proba[0] * 300
                if(rescale_proba > 1):

                    coreEntities.append(name)
                    count += 1
                    # print(name, rescale_proba)

                if count == 3:
                    break

            # e_time = time.clock()
            # print("1# time consuming: %f" %(e_time - s_time))


            # s_time = time.clock()
            # predict emotion of core entity
            for entity in coreEntities:
                # text = news['title'] + '\n' + news['content']
                # relatedSents = []
                # for sent in re.split(r'[\n\t，。！？“”（）]', text):
                #     if (entity in sent):
                #         relatedSents.append(sent)
                # relatedText = ' '.join(relatedSents)
                # emotionTfIdfFeature = self.emotionTfIdf.transform([relatedText]).toarray()
                # emotion = self.emotionCLF.predict(emotionTfIdfFeature)
                # predictCoreEntityEmotion[entity] = emotion[0]

                predictCoreEntityEmotion[entity] = "POS"

            # e_time = time.clock()
            # print("#2 time consuming: %f" % (e_time - s_time))

            all_entities = []
            all_emotions = []
            for entity, emotion in predictCoreEntityEmotion.items():
                all_entities.append(entity.replace('\t', '').replace('\n', '').replace(',', ''))
                all_emotions.append(emotion)

            f_submit.write(news['newsId'])
            f_submit.write('\t')
            f_submit.write(','.join(all_entities))
            f_submit.write('\t')
            f_submit.write(','.join(all_emotions))
            f_submit.write('\n')

def main():
    trainer = LR_ent()
    trainer.prepare_data()
    trainer.fit(type="MS")
    trainer.trainEmotion(type="MS")
    trainer.predict_score(type="MS")

    # trainer.fit()
    # trainer.predict()

if __name__ == '__main__':
    main()