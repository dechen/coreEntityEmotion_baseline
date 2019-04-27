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

jieba.load_userdict('./models/nerDict.txt') #为结巴分词指定自定义词典

class NN_ent(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.train_dt = None
        self.test_dt = None

    def fit(self):
        pass

    def prepare_data(self):
        pass