#coding:utf-8

import os
from abc import ABCMeta, abstractmethod
import time
import tqdm

from sklearn.model_selection import KFold

class BaseModel():
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    def prepare_data(self):
        """
        prepare data for current model
        """
        self.preprocessing()
        pass

    @abstractmethod
    def preprocessing(self):
        pass

    def split_train_eval_data(self, data, n_splits=5):
        """
        将训练样本划分为训练集和验证集
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        tmp = kf.split(data)
        train_idx, eval_idx = next(tmp)

        ###生成训练集和验证集
        train_dt = []
        eval_dt = []
        for idx in train_idx:
            train_dt.append(data[idx])

        for idx in eval_idx:
            eval_dt.append(data[idx])

        return train_dt, eval_dt

    def predict_proba(self):
        pass

    def predict(self):
        pass

    def predict_score(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()