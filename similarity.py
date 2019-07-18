from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import json
from flair.embeddings import ELMoEmbeddings
from avg_embeddings import run_context_avg_benchmark
import scipy
from sklearn.metrics import mean_squared_error
from math import sqrt
from flair.data import Sentence
import functools as ft
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

class Loader(object):

    def __init__(self, model_name, path, files):
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
        return [ pd.DataFrame(x) for x in L ]


class Elmo(object):
    def __init__(self, model_name):
        self.model = ELMoEmbeddings(model_name)
        self.train = None
        self.test = None
        self.train_sims = None
        self.test_sims = None
        self.results = None

    def similarity(self, df, func, label_a, label_b):
        sentences1 = [Sentence(s) for s in df[label_a]]
        sentences2 = [Sentence(s) for s in df[label_b]]
        benchmark = ft.partial(run_context_avg_benchmark, model=self.model)
        sims = benchmark(sentences1, sentences2)
        return sims

    def get_sims(self, func):
        self.train_sims = self.similarity(self.train, func, 't', 'h')
        self.test_sims = self.similarity(self.test, func, 't', 'h')
        return self

    def format(self):
        features_train = np.array(self.train_sims).reshape(1,-1)
        results_train = self.train['similarity'].values.reshape(1, -1)
        features_test = np.array(self.test_sims).reshape(1, -1)
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
        x = results['features_train']
        y = results['results_train']
        test = ['features_test']
        l_reg = LinearRegression()
        l_reg.fit(x, y)
        test_predict = l_reg.predict(test)
        return test_predict.flatten()      


class ElmoRegression(object):

    def __init__(self, model_name, path, files):
        self.model = ELMoEmbeddings(model_name)
        self.path = path
        self.files = files
        self.test, self.train = self.load_dataset()

    def get_sims(self, df, label_a, label_b):
        sentences1 = [Sentence(s) for s in df[label_a]]
        sentences2 = [Sentence(s) for s in df[label_b]]
        benchmark = ft.partial(run_context_avg_benchmark, model=self.model)
        sims = benchmark(sentences1, sentences2)
        return sims
    
    def evaluate_testset(self, x, y, test):
        l_reg = LinearRegression()
        l_reg.fit(x, y)
        test_predict = l_reg.predict(test)
        return test_predict

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
        return [ pd.DataFrame(x) for x in L ]

    def regression(self):
        features_train = np.array(self.get_sims(self.train, 't', 'h')).reshape(1,-1)
        results_train = self.train['similarity'].values.reshape(1, -1)
        features_test = np.array(self.get_sims(self.test, 't', 'h')).reshape(1, -1)
        print(features_train.shape)
        print(results_train.shape)
        print(features_test.shape)
        test_results = self.evaluate_testset(features_train, results_train, features_test)
        return test_results

if __name__ == '__main__':
    model_name = 'pt'
    PATH = './datasets/ASSIN/'
    files = {
        'train' : 'assin-ptbr-train-hartmann.json',
        'test' : 'assin-ptbr-test-gold-hartmann.json'
    }
    model = ElmoRegression(model_name, PATH, files)
    test_results = model.regression().flatten()
    gold = model.test['similarity'].astype(float).values.flatten()
    print('GOLD', gold)
    print('RESULTS', test_results)
    pearson_correlation = pearsonr(gold, test_results)[0]
    mse = mean_squared_error(gold, test_results)
    print("Pearson Correlation: {0} \n MSE: {1} \n".format(pearson_correlation, mse))