from similarity import Loader, Regression, Embedding
from distance import flair_distance, gensim_distance
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from flair.embeddings import ELMoEmbeddings

import json
from loguru import logger

logger.add("evaluate_{time}.log")

model_name = 'pt'
PATH = './datasets/ASSIN/'
files = {
    'train' : 'assin-ptbr-train-hartmann.json',
    'test' : 'assin-ptbr-test-gold-hartmann.json'
}
EMBEDDINGS_DIR = './embeddings/USP/'
wang2vec_file = 'cbow_s600.txt'
wang2vec_fname = 'wang2vec.model'

test, train = Loader(PATH, files).load_dataset()
gold = test['similarity'].astype(float).values.flatten()

def run_NILC(model, func, dst):
    test_results = run_regression(model, func)
    model_name = '_'.join(dst.rstrip('.model').split('/')[-2:-1])
    measure = {
        'model' : model_name
        'pearson' : pearsonr(gold, test_results)[0],
        'MSE' : mean_squared_error(gold, test_results)
        }
    return measure

def run_regression(model, func):
    model.test = test
    model.train = train
    model.get_sims(func).format()
    LR = Regression(model.results)
    test_results = LR.evaluate_testset()
    return test_results 
    measure = {
        'model' : 'ELMo'
        'pearson' : pearsonr(gold, test_results)[0],
        'MSE' : mean_squared_error(gold, test_results)
        }
    return measure

def run_elmo(lang, func):
    model = Embedding(ELMoEmbeddings(lang))
    test_results = run_regression(model, func)


stats = []
for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
    for name in files:
        dst = path + name
        if name.endswith('.model'):
            embedding = KeyedVectors.load(dst)
            model = Embedding(embedding)
            measure = run_NILC(model, gensim_distance)
            logger.debug(measure)
            stats.append(measure)


elmo = Embedding(ELMoEmbeddings('pt'))
measure = run_model(elmo, flair_distance)
stats.append(measure)


stats = {
    'model' : 'ELMO-SUM',
    'pearson' : pearsonr(gold, test_results)[0],
    'MSE' : mean_squared_error(gold, test_results)
}

stats_table = pd.DataFrame(stats, index=[0])

with open('stats-' + str(int(datetime.datetime.now().timestamp())) + '.json', 'w+') as f:
    json.dump(stats, f)




loaded_wang2vec_model = KeyedVectors.load(EMBEDDINGS_DIR + wang2vec_fname)
wang2vec = Embedding(loaded_wang2vec_model)
wang2vec.test = test
wang2vec.train = train
wang2vec.get_sims(gensim_distance).format()
LR = Regression(wang2vec.results)
test_results = LR.evaluate_testset()

stats = {
    'model' : 'WANG2VEC-CBOW-600-SUM',
    'pearson' : pearsonr(gold, test_results)[0],
    'MSE' : mean_squared_error(gold, test_results)
}

print(stats)