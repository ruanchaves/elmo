import pandas as pd
import numpy as np
import json
import scipy
from math import sqrt
import functools as ft
import datetime
import sys

from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from distance import flair_distance

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Loader(object):

    def __init__(self, path, files):
        self.path = path
        self.files = files
        self.test, self.train = self.load_dataset()

    def load_dataset(self):
        L = []
        for key in sorted(tuple(self.files)):
            fname = self.path + self.files[key]
            with open(fname, 'r') as f:
                data = json.load(f)
                if key == 'test':
                    max_len = len(data)
                L.append(data)
        for idx,item in enumerate(L):
            L[idx] = item[0:max_len]
        return ( pd.DataFrame(x) for x in L )

class Embedding(object):

    def __init__(self, model=None):
        self.model = model
        self.train = None
        self.test = None
        self.train_sims = None
        self.test_sims = None
        self.results = None

    def similarity(self, df, func, label_a, label_b):
        sentences1 = [Sentence(' '.join(s)) for s in df[label_a]]
        sentences2 = [Sentence(' '.join(s)) for s in df[label_b]]
        benchmark = ft.partial(func, model=self.model)
        sims = benchmark(sentences1, sentences2)
        return sims

    def get_sims(self, func):
        self.train_sims = self.similarity(self.train, func, 't', 'h')
        self.test_sims = self.similarity(self.test, func, 't', 'h')
        return self

    def format(self):
        features_train = np.array(self.train_sims).reshape(-1,1)
        results_train = self.train['similarity'].values.astype(float).flatten()
        features_test = np.array(self.test_sims).reshape(-1, 1)
        self.results = {
            'features_train' : features_train,
            'results_train' : results_train,
            'features_test' : features_test
        }
        return self

class Regression(object):

    def __init__(self, results):
        self.results = results

    def evaluate_testset(self):
        x = self.results['features_train']
        y = self.results['results_train']
        print(x)
        print(y)
        test = self.results['features_test']
        print(test)
        l_reg = LinearRegression()
        l_reg.fit(x, y)
        test_predict = l_reg.predict(test)
        print(test_predict)
        return test_predict.flatten()      

if __name__ == '__main__':
    pass