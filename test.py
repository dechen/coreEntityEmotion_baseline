# -*- coding: utf-8 -*-
import sys, getopt
import time
from joblib import load
from train import Train
import re
import codecs
import jieba
from tqdm import tqdm

jieba.load_userdict('./models/nerDict.txt') #为结巴分词指定自定义词典

class Test(Train):
    def __init__(self):
        super(Test, self).__init__()
        self.coreEntityTfIdf = load('models/nerTfIdf.joblib')
        self.coreEntityCLF = load('models/coreEntityCLF.joblib')
        self.featureName = self.coreEntityTfIdf.get_feature_names()

        # self.emotionTfIdf = load('models/emotionTfIdf.joblib')
        # self.emotionCLF = load('models/emotionCLF.joblib')

    def testCoreEntity(self):
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


def parseArgv():
    ###解析命令行
    argv = sys.argv[1:]
    global DATA_SAVE_PATH
    try:
        opts, args = getopt.getopt(argv, "h", ["dfile="])
    except getopt.GetoptError:
        print("src.py --data <datafile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("src.py --data <datafile>")
        elif opt == "--dfile":
            DATA_SAVE_PATH = arg

    assert ((DATA_SAVE_PATH != ""))

if __name__ == '__main__':
    # parseArgv()

    ###训练
    # trainer = Train()
    # trainer.trainCoreEntity()

    ###predict
    test = Test()
    test.testCoreEntity()


