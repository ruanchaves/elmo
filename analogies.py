import yaml
import os
from gensim.models import KeyedVectors
import sys
from loguru import logger
import json
import datetime

if __name__ == '__main__':

    logger.add("analogies_{time}.log")
    logger.debug("START")

    settings = {}
    with open("settings.yaml", 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    EMBEDDINGS_DIR = settings['NILC']['dir']
    ANALOGIES_DIR = settings['NILC']['analogies']

    analogies_file = 'analogies.json'

    try:
        open(analogies_file, 'r').close()
    except:
        with open(analogies_file,'w+') as f:
            json.dump([], f)

    stats = {}
    for path, subdirs, files in os.walk(ANALOGIES_DIR):
        for name in files:
            dst = path + '/' + name
            for path2, subdirs2, files2 in os.walk(EMBEDDINGS_DIR):
                for name2 in files2:
                    dst2 = path2 + '/' + name2
                    if name2.endswith('.model'):
                        key = name.rstrip('.txt') + '_' + path2.split('/')[-1] + '_' + name2.rstrip('.model') 
                        with open(analogies_file,'r') as f:
                            stats = json.load(f)
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