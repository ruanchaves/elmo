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

if __name__ == '__main__':
    
    # logger.add("evaluate_{time}.log")

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

    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
        for name in files:
            dst = path + '/' + name
            if name.endswith('.model'):
                embedding = KeyedVectors.load(dst)
                print(embedding.wv['unk'])

    # with open('stats-' + str(int(datetime.datetime.now().timestamp())) + '.json', 'w+') as f:
    #     json.dump(stats, f)