from similarity import Loader, Regression, Embedding, CombinedEmbedding
from distance import flair_distance, gensim_distance, combined_distance
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from flair.embeddings import ELMoEmbeddings
import json
from loguru import logger
import yaml
import sys
import os
import datetime

def run_NILC(model, func, dst):
    test_results = run_regression(model, func)
    model_name_tokens = dst.rstrip('.model').split('/')
    model_name = model_name_tokens[-2] + '_' + model_name_tokens[-1]
    measure = {
        'model' : model_name,
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

def run_elmo(model, func):
    test_results = run_regression(model, func)
    measure = {
        'model' : 'ELMo',
        'pearson' : pearsonr(gold, test_results)[0],
        'MSE' : mean_squared_error(gold, test_results)
        }
    return measure

def run_combined(model, func, dst):
    test_results = run_regression(model, func)
    model_name_tokens = dst.rstrip('.model').split('/')
    model_name = model_name_tokens[-2] + '_' + model_name_tokens[-1]
    measure = {
        'model' : "ELMo_" + model_name,
        'pearson' : pearsonr(gold, test_results)[0],
        'MSE' : mean_squared_error(gold, test_results)
        }
    return measure

if __name__ == '__main__':


    stats = []    
    settings = {}
    with open("settings.yaml", 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    model_name = settings['ELMo']['name']
    PATH = settings['ASSIN']['path']
    files = settings['ASSIN']['files']['ptbr']
    EMBEDDINGS_DIR = settings['NILC']['dir']
    LOGS_PATH = settings['logs']['path']
    RESULTS_PATH = settings['results']['path']

    logger.add(LOGS_PATH + "evaluate_{time}.log")

    test, train = Loader(PATH, files).load_dataset()
    gold = test['similarity'].astype(float).values.flatten()

    elmo = Embedding(ELMoEmbeddings('pt'))
    measure = run_elmo(elmo, flair_distance)
    logger.debug(measure)
    stats.append(measure)

        for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
            for name in files:
                dst = path + '/' + name
                if name.endswith('.model'):
                    embedding = KeyedVectors.load(dst)
                    model = Embedding(embedding)
                    measure = run_NILC(model, gensim_distance, dst)
                    logger.debug(measure)
                    stats.append(measure)

    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
        for name in files:
            dst = path + '/' + name
            if name.endswith('.model'):
                gensim_model = KeyedVectors.load(dst)
                flair_model = Embedding(ELMoEmbeddings('pt'))
                model = CombinedEmbedding(gensim_model=gensim_model, flair_model=flair_model)
                measure = run_combined(model, combined_distance, dst)
                logger.debug(measure)
                stats.append(measure)

    stats_fname = RESULTS_PATH + 'stats-' + str(int(datetime.datetime.now().timestamp())) + '.json'
    with open(stats_fname, 'w+') as f:
        json.dump(stats, f)
        logger.debug(stats_fname)