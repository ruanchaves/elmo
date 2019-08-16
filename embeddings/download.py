import zipfile
import shutil
import os
import multiprocessing
import json
from urllib.request import urlopen
from loguru import logger
from gensim.models import KeyedVectors

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

def unzip(target):
    with zipfile.ZipFile(target['source'],"r") as zip_ref:
        zip_ref.extractall(target['path'])

def gensim_load_model(item):
    logger.debug(item['source'] + ' -> ' + item['destination'])
    model = KeyedVectors.load_word2vec_format(item['source'])
    model.save(item['destination'])

if __name__ == '__main__':

    # logger.add("download_{time}.log")
    NILC_DIR = './NILC/'
    with open("links.json","r") as f:
        links = json.load(f)

    for url in links:

        folder = url.split('/')[-2]
        dst = url.split('/')[-1]

        try:
            os.mkdir(NILC_DIR + folder)
        except FileExistsError as e:
            pass

        destination = NILC_DIR + folder + '/' + dst

        if os.path.isfile(destination):
            continue

        logger.debug('Downloading {0}'.format(url))

        get_large_file(url, destination)

    targets = []
    for path, subdirs, files in os.walk(NILC_DIR):
        for name in files:
            if name.endswith('.zip'):
                source = path + '/' + name
                result = name.rstrip('.zip') + '.txt'
                result_dst = path + '/' + result
                if os.path.isfile(result_dst):
                    continue
                targets.append({
                    'source' : source,
                    'path': path
                })

    if targets:
        p = multiprocessing.Pool()
        p.map(unzip, targets)
        p.close()
        p.join()

    targets = []
    for path, subdirs, files in os.walk(NILC_DIR):
        for name in files:
            if name.endswith('.txt'):
                source = path + '/' + name
                destination = source.rstrip('.txt') + '.model'
                if os.path.isfile(destination):
                    continue
                targets.append({
                    'source' : source,
                    'destination' : destination
                })
    
    cores = 4
    if targets:
        p = multiprocessing.Pool(cores)
        p.map(gensim_load_model, targets)
        p.close()
        p.join()