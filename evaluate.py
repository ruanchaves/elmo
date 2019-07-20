from similarity import Loader, Regression, Embedding
from distance import cosine_distance
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


def run_regression(model, test, train, func):
    model.test = test
    model.train = train
    model.get_sims(func).format()
    LR = Regression(model.results)
    test_results = LR.evaluate_testset()
    return test_results

def run_test(model, func, gold, test, train, name):
    test_results = run_regression(model, test, train, func)
    return {
        'test': name
        'pearson': pearsonr(gold, test_results)[0]
        'MSE': mean_squared_error(gold, test_results)
    }

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def load_ASSIN(lang):
    files = settings['ASSIN']['files'][lang]
    PATH = settings['ASSIN']['path']
    test, train = Loader(PATH, files).load_dataset()
    gold = test['similarity'].astype(float).values.flatten()
    return gold, test, train

def get_NILC(EMBEDDINGS_DIR):
    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
            for name in files:
                dst = path + '/' + name
                if name.endswith('.model'):
                    yield dst

def get_measure(model, LANGUAGE, test_name):
        gold, test, train = load_ASSIN(LANGUAGE)
        measure = run_test(model, cosine_distance, test, train, gold, test_name)
        measure['lang'] = LANGUAGE   
        return measure

def save_results(fname, results):
    with open(fname,'w+') as f:
        json.dump(results, fname)

def get_analogies(ANALOGIES_DIR, EMBEDDINGS_DIR):
    for path, subdirs, files in os.walk(ANALOGIES_DIR):
        for name in files:
            dst = path + '/' + name
            for path2, subdirs2, files2 in os.walk(EMBEDDINGS_DIR):
                for name2 in files2:
                    dst2 = path2 + '/' + name2
                    if name2.endswith('.model'):
                        yield path, path2, name, name2, dst, dst2

if __name__ == '__main__':

    stats = []    
    settings = {}
    tests = {}
    results = []

    settings = load_yaml("settings.yaml")
    tests = load_yaml("test.yaml")

    EMBEDDINGS_DIR = settings['NILC']['dir']
    ANALOGIES_DIR = settings['NILC']['analogies']
    ANALOGIES_FILE = RESULTS_PATH + 'analogies.json'

    LOGS_PATH = settings['logs']['path']
    RESULTS_PATH = settings['results']['path']

    logger.add(LOGS_PATH + "evaluate_{time}.log")
    RESULTS_FILE = RESULTS_PATH + 'stats-' + str(int(datetime.datetime.now().timestamp())) + '.json'

    if tests['ELMo']:
        test_name = 'ELMo'
        model = Embedding(flair_model=ELMoEmbeddings('pt'))

        LANGUAGE = 'ptbr'
        measure = get_measure(model, LANGUAGE, test_name)
        logger.debug(measure)
        results.append(measure)

        LANGUAGE = 'pteu'
        measure = get_measure(model, LANGUAGE, test_name)
        logger.debug(measure)
        results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['NILC']:
        test_name = 'NILC'
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['NILC_with_unk']:
        test_name = 'NILC_with_unk'
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, unk=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['ELMo_NILC']:
        test_name = 'ELMo_NILC'
        elmo = Embedding(flair_model=ELMoEmbeddings('pt'))
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, flair_model=elmo, unk=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['ELMO-SIF']:
        test_name = 'ELMO-SIF'
        elmo = Embedding(flair_model=ELMoEmbeddings('pt'))
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, flair_model=elmo, flair_sif=True, unk=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['NILC-SIF']:
        test_name = 'NILC-SIF'
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, gensim_sif=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['ELMo-SIF_NILC-SIF']:
        test_name = 'ELMo-SIF_NILC-SIF'
        elmo = Embedding(flair_model=ELMoEmbeddings('pt'))
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, flair_model=elmo, gensim_sif=True, flair_sif=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['ELMO-SIF_NILC']:
        test_name = 'ELMO-SIF_NILC'
        elmo = Embedding(flair_model=ELMoEmbeddings('pt'))
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, flair_model=elmo, gensim_sif=False, flair_sif=True)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['ELMo_NILC-SIF']:
        test_name = 'ELMo_NILC-SIF'
        elmo = Embedding(flair_model=ELMoEmbeddings('pt'))
        for fname in get_NILC(EMBEDDINGS_DIR):
            emb = KeyedVectors.load(fname)
            model = Embedding(gensim_model=emb, flair_model=elmo, gensim_sif=True, flair_sif=False)

            LANGUAGE = 'ptbr'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

            LANGUAGE = 'pteu'
            measure = get_measure(model, LANGUAGE, test_name)
            logger.debug(measure)
            results.append(measure)

    save_results(RESULTS_FILE, results)

    if tests['analogies']:
        test_name = 'analogies'
        try:
            open(ANALOGIES_FILE,'r').close()
        except:
            with open(ANALOGIES_FILE,'w+') as f:
                json.dump({}, f)
        for path, path2, name, name2, dst, dst2 in get_analogies(ANALOGIES_DIR, EMBEDDINGS_DIR):
            key = name.rstrip('.txt') + '_' + path2.split('/')[-1] + '_' + name2.rstrip('.model') 
            with open(ANALOGIES_FILE,'r') as f:
                stats = json.load(ANALOGIES_FILE)
            try:
                stats[key]
                continue
            except:
                pass
            embedding = KeyedVectors.load(dst2)
            score = embedding.evaluate_word_analogies(dst)[0]
            stats[key] = score
            with open(analogies_file,'w+') as f:
                json.dump(stats, f)
            logger.debug(key, stats)