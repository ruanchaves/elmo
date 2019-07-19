import json
import subprocess
import shutil
from urllib.request import urlopen
import sys
import os
import subprocess
from loguru import logger
import yaml
import subprocess
import multiprocessing
from gensim.models import KeyedVectors
import psutil
from time import sleep

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

def unzip(cmd):
    return subprocess.call(cmd)

def gensim_load_model(item):
    while True:
        mem = psutil.virtual_memory().percent
        if mem > item['limit']:
            sleep(item['wait_interval'])
        else:
            break
    model = KeyedVectors.load_word2vec_format(item['dst'])
    model.save(item['destination'])

if __name__ == '__main__':
    settings = {}
    
    with open("settings.yaml", 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    logger.add("download_{time}.log")

    NILC_DIR = settings['NILC']['dir']
    links = settings['NILC']['sources']

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
                dst = path + '/' + name
                result = name.rstrip('.zip') + '.txt'
                result_dst = path + '/' + result
                if os.path.isfile(result_dst):
                    continue
                targets.append({
                    'dst' : dst,
                    'path': path
                })

    if targets:
        cmds_list = [ ['unzip','-o',item['dst'], '-d', item['path']] for item in targets ] 
        p = multiprocessing.Pool(settings['unzip']['cores'])
        p.map(unzip, cmds_list)
        p.close()
        p.join()

    targets = []

    limit = settings['gensim']['memory_percent_limit']
    wait_interval = settings['gensim']['wait_interval']
    for path, subdirs, files in os.walk(NILC_DIR):
        for name in files:
            if name.endswith('.txt'):
                dst = path + '/' + name
                destination = dst.rstrip('.txt') + '.model'
                if os.path.isfile(destination):
                    continue
                targets.append({
                    'dst' : dst,
                    'destination' : destination,
                    'limit': limit,
                    'wait_interval': wait_interval
                })
    
    # if targets:
    #     p = multiprocessing.Pool(settings['gensim']['cores'])
    #     p.map(gensim_load_model, targets)
    #     p.close()
    #     p.join()

    for idx, item in enumerate(targets):
        gensim_load_model(item)