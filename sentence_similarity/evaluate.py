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
from wordfreq import word_frequency
import random
import concurrent.futures
from bert_serving.client import BertClient

def run_regression(model, func):
    model.get_sims(func).format()
    LR = Regression(model.results)
    test_results = LR.evaluate_testset()
    return test_results

def run_test(model, func, name):
    test_results = run_regression(model, func)
    return {
        'test': name,
        'pearson': pearsonr(model.gold, test_results)[0],
        'MSE': mean_squared_error(model.gold, test_results)
    }

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def load_ASSIN(lang, sick=False):
    if sick:
        files = settings['SICK']['files'][lang]
        PATH = settings['SICK']['path']
    else:
        files = settings['ASSIN']['files'][lang]
        PATH = settings['ASSIN']['path']
    test, train = Loader(PATH, files).load_dataset()
    gold = test['similarity'].astype(float).values.flatten()
    return gold, test, train

def get_measure(model, LANGUAGE, test_name, sick=False):
        gold, test, train = load_ASSIN(LANGUAGE, sick=sick)
        model.test = test
        model.train = train
        model.gold = gold
        measure = run_test(model, cosine_distance, test_name)
        measure['lang'] = LANGUAGE
        measure['timestamp'] = int(datetime.datetime.now().timestamp())   
        return measure

def get_NILC(EMBEDDINGS_DIR):
    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
            for name in files:
                dst = path + '/' + name
                if name.endswith('.model'):
                    yield dst
                    
def save_results(fname, results):
    with open(fname,'w+') as f:
        json.dump(results, f)

def get_analogies(ANALOGIES_DIR, EMBEDDINGS_DIR):
    for path, subdirs, files in os.walk(ANALOGIES_DIR):
        for name in files:
            dst = path + '/' + name
            for path2, subdirs2, files2 in os.walk(EMBEDDINGS_DIR):
                for name2 in files2:
                    dst2 = path2 + '/' + name2
                    if name2.endswith('.model'):
                        yield path, path2, name, name2, dst, dst2

def test_already_done(name, lang):
    with open(RESULTS_FILE,'r') as f:
        results = json.load(f)
        for item in results:
            if item['test'] == name and item['lang'] == lang:
                logger.debug("SKIP - {0}".format(name))
                return True
    logger.debug("RUN - {0}".format(name))
    return False

class WordFreq(object):
    def __getitem__(self, key):
        return word_frequency(key, 'pt', wordlist='best', minimum=0.0)

class WordFreqEn(object):
    def __getitem__(self, key):
        return word_frequency(key, 'en', wordlist='best', minimum=0.0)


def call_test(skip_list=[], test_name="", langs=[], template="", params={}, ANALOGIES_FILE="", ANALOGIES_DIR="", EMBEDDINGS_DIR=""):
 
    if template == 'sick':
        for lang in langs:
            if test_already_done(test_name, lang):
                continue
            params["flair_model"] = ELMoEmbeddings('original')
            model = Embedding(**params)
            measure = get_measure(model, 'en', test_name, sick=True)
            yield measure

    elif template == 'bert':
        for lang in langs:
            if test_already_done(test_name, lang):
                continue
            params["bert_model"] = BertClient()
            model = Embedding(**params)
            measure = get_measure(model, lang, test_name)
            yield measure            

    elif template == 'flair':
        for lang in langs:
            if test_already_done(test_name, lang):
                continue
            params["flair_model"] = ELMoEmbeddings('pt')
            model = Embedding(**params)
            measure = get_measure(model, lang, test_name)
            yield measure            

    elif template == 'flair-custom-1' or template == 'flair-custom-2':
        for lang in langs:
            if test_already_done(test_name, lang):
                continue
            if template == 'flair-custom-1':
                params["flair_model"] = ELMoEmbeddings(options_file="../embeddings/elmo/options.json", weight_file="../embeddings/elmo/elmo_pt_weights.hdf5")
            elif template == 'flair-custom-2':
                params["flair_model"] = ELMoEmbeddings(options_file="../embeddings/elmo/options_dgx1.json", weight_file="../embeddings/elmo/elmo_pt_weights_dgx1.hdf5")
            model = Embedding(**params)
            measure = get_measure(model, lang, test_name)
            yield measure         

    elif template == 'gensim' or template == 'flair-gensim' or template == 'custom-flair-gensim-1' or template == 'custom-flair-gensim-2':
        assert(EMBEDDINGS_DIR != None)
        for fname in get_NILC(EMBEDDINGS_DIR):
            for item in skip_list:
                if item in fname:
                    break
            else:
                t = test_name + '_' + fname
                for lang in langs:
                    if test_already_done(t, lang):
                        continue
                    emb = KeyedVectors.load(fname)
                    params["gensim_model"] = emb
                    if template == 'flair-gensim':
                        params["flair_model"] = ELMoEmbeddings('pt')
                    elif template == 'custom-flair-gensim-1':
                        params["flair_model"] = ELMoEmbeddings(options_file="../embeddings/elmo/options.json", weight_file="../embeddings/elmo/elmo_pt_weights.hdf5")
                    elif template == 'custom-flair-gensim-2':
                        params["flair_model"] = ELMoEmbeddings(options_file="../embeddings/elmo/options_dgx1.json", weight_file="../embeddings/elmo/elmo_pt_weights_dgx1.hdf5")
                    model = Embedding(**params)
                    measure = get_measure(model, lang, t)
                    yield measure
        
    elif template == "analogies":
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
            with open(ANALOGIES_FILE,'w+') as f:
                json.dump(stats, f)
            message = {
                "key" : key,
                "stats": stats[key]
            }
            yield message

def evaluate_sentence_similarity(parameters):
    parameters["EMBEDDINGS_DIR"] = EMBEDDINGS_DIR
    parameters["ANALOGIES_FILE"] = ANALOGIES_FILE
    parameters["ANALOGIES_DIR"] = ANALOGIES_DIR
    class_name = parameters["params"]["freqs"]
    parameters["params"]["freqs"] = globals()[class_name]()

    for measure in call_test(**parameters):
        yield measure

if __name__ == '__main__':
  
    settings = {}
    tests = {}
    results = []

    test_file = '../settings/' + os.environ['TESTS']
    stats_file = os.environ['RESULTS']

    settings = load_yaml("settings.yaml")
    tests = load_yaml(test_file)

    LOGS_PATH = settings['logs']['path']
    RESULTS_PATH = settings['results']['path']

    EMBEDDINGS_DIR = settings['NILC']['dir']
    ANALOGIES_DIR = settings['NILC']['analogies']
    ANALOGIES_FILE = RESULTS_PATH + 'analogies.json'

    freqs = WordFreq()

    logger.add(LOGS_PATH + "evaluate_{time}.log")
    RESULTS_FILE = RESULTS_PATH + stats_file
    try:
        open(RESULTS_FILE, 'r').close()
    except:
        with open(RESULTS_FILE, 'w+') as f:
            json.dump([], f)

    training_list = []
    for idx,key in enumerate(tests):
        if key in tests['queue']:
            parameters = tests[key]
            training_list.append(parameters)

    logger.debug('Start evaluation.')
    logger.debug(RESULTS_FILE)
    results = []
    # random.shuffle(training_list)
    for item in training_list:
        for measure in evaluate_sentence_similarity(item):
            with open(RESULTS_FILE, 'r') as f:
                results = json.load(f)
            logger.debug(measure)
            results.append(measure)
            with open(RESULTS_FILE, 'w+') as f:
                json.dump(results, f)