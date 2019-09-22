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

    def __init__(self, gensim_model=None, flair_model=None, bert_model=None, gensim_sif=False, flair_sif=False, freqs={}, a=0.001, unk=False, total_freq=1.0):
        self.gensim_model = gensim_model
        self.flair_model = flair_model
        self.bert_model = bert_model
        self.gensim_sif = gensim_sif
        self.flair_sif = flair_sif
        self.freqs = freqs
        self.total_freq = total_freq
        self.a = a
        self.train = None
        self.test = None
        self.gold = None
        self.train_sims = None
        self.test_sims = None
        self.results = None
        self.unk = unk
        with open('./sources/dictionary.json','r') as f:
            self.dictionary = json.load(f)

    def similarity(self, df, func, label_a, label_b):
        sentences1 = [ s for s in df[label_a]]
        sentences2 = [ s for s in df[label_b]]
        benchmark = ft.partial(func, gensim_model=self.gensim_model, bert_model=self.bert_model, flair_model=self.flair_model, gensim_sif=self.gensim_sif, flair_sif=self.flair_sif, freqs=self.freqs, total_freq=self.total_freq, a=self.a, unk=self.unk, dictionary=self.dictionary)
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
        test = self.results['features_test']
        l_reg = LinearRegression()
        l_reg.fit(x, y)
        test_predict = l_reg.predict(test)
        return test_predict.flatten()      

if __name__ == '__main__':
    pass