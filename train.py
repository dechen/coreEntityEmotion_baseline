# -*- coding: utf-8 -*-

import os
import time
import jieba
import codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from joblib import dump, load
import re
from scipy.sparse import vstack
from tqdm import tqdm

class Train():
    def __init__(self):
        # load nerDict as named entity recognizer
        self.loadNerDict()

    def trainCoreEntity(self):
        '''
        train model for coreEntity
        Baseline use entityDict for named entity recognition, you can use a more wise method.
        Baseline use tfIdf score as feature and LR as classification model
        :return:
        '''
        # 1. train tfIdf as core entity score model
        trainData = self.loadData('data/coreEntityEmotion_train.txt')

        print("loading all ner corpus from train data...")

        if not os.path.exists('models/nerCorpus.joblib'):
            nerCorpus = []
            for news in tqdm(trainData):
                nerCorpus.append(' '.join(self.getEntity(news)))
            dump(nerCorpus, 'models/nerCorpus.joblib')
        else:
            nerCorpus = load('models/nerCorpus.joblib')

        if not os.path.exists("models/nerTfIdf.joblib"):
            print("fitting ner tfIdf model...")
            tfIdf = TfidfVectorizer()
            tfIdf.fit(nerCorpus)
            # 1.1 save tfIdf model
            dump(tfIdf, 'models/nerTfIdf.joblib')
        else:
            print("loading ner tfIdf model from file...")
            tfIdf = load('models/nerTfIdf.joblib')

        if not os.path.exists('models/CoreEntityCLF.joblib'):
            # 2. train LR with tfIdf score as features
            isCoreX = []
            isCoreY = []
            # for news in tqdm(trainData[:5]):
            for news in tqdm(trainData):

                tfIdfNameScore = self.getTfIdfScore(news, tfIdf)

                coreEntity_GroundTruth = [x['entity'] for x in news['coreEntityEmotions']]
                for name, score in tfIdfNameScore:
                    if(name in coreEntity_GroundTruth):
                        isCoreX.append([score])
                        isCoreY.append(1)
                    else:
                        isCoreX.append([score])
                        isCoreY.append(0)

            dump(isCoreX, 'models/isCoreX.joblib')
            dump(isCoreY, 'models/isCoreY.joblib')
            # print(isCoreX, isCoreY)
            # return

            # 3. train LR model for coreEntity

            print("training LR model for coreEntity...")
            clf = LogisticRegression(random_state=0, solver='lbfgs',
                                     multi_class='multinomial').fit(isCoreX, isCoreY)
            dump(clf, 'models/CoreEntityCLF.joblib')
        else:
            print("loading LR model for file...")
            clf = load('models/CoreEntityCLF.joblib')

    def trainEmotion(self):
        '''
        train emotion model
        Baseline use tfIdf vector as feature, linearSVC as classfication model
        :return:
        '''
        trainData = self.loadData('data/coreEntityEmotion_train.txt')

        if not os.path.exists('models/emotionX.joblib'):

            emotionX = []
            emotionY = []

            print("loading emotion corpus from train data...")

            # 1. get all related sentences to the entities
            for news in tqdm(trainData):

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
            for news in trainData:
                emotionWordCorpus.append(' '.join(self.getWords(news)))

            tfIdf = TfidfVectorizer()
            tfIdf.fit(emotionWordCorpus)
            dump(tfIdf, 'models/emotionTfIdf.joblib')
        else:
            print("loading emotion tfIdf model...")
            tfIdf = load('models/emotionTfIdf.joblib')

        if not os.path.exists('models/emotionCLF.joblib'):

            # 3. use naive bayes to train emotion classifiction
            s_time = time.clock()
            lst = [tfIdf.transform(x) for x in emotionX]
            e_time = time.clock()
            print("time consuming: %f" %(e_time - s_time))

            emotionX = vstack(lst).tocsr()

            print("training emotion clf with linearSVC...")

            print(emotionX.shape)
            clf = MultinomialNB()
            clf.fit(emotionX, emotionY)

            print(clf.score(emotionX, emotionY))

            dump(clf, 'models/emotionCLF.joblib')
        else:
            clf = load('models/emotionCLF.joblib')

    def getTfIdfScore(self, news, tfIdf):
        featureName = tfIdf.get_feature_names()

        doc = self.getEntity(news)

        tfIdfFeatures = tfIdf.transform([' '.join(doc)])

        tfIdfScores = tfIdfFeatures.data
        # normalize
        tfIdfScoresNorm = normalize([tfIdfScores], norm='max')

        tfIdfNameScore = [(featureName[x[0]], x[1]) for x in zip(tfIdfFeatures.indices, tfIdfScoresNorm[0])]
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

    def loadData(self, filePath):
        f = codecs.open(filePath,'r', 'utf-8')
        data = []
        for line in f.readlines():
            news = json.loads(line.strip())
            data.append(news)
        return data

if __name__ == '__main__':
    trainer = Train()
    trainer.trainCoreEntity()
    # trainer.trainEmotion()
