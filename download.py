import json
import subprocess
import shutil
from urllib.request import urlopen
import sys
import os
import subprocess
from loguru import logger

logger.add("file_{time}.log")

filenames = 'embeddings.txt'
DIR = './embeddings/USP/'

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

with open(filenames,'r') as f: 
    links = f.read().split('\n')
    links = list(filter(lambda x: x.strip() , links))

for url in links:
    folder = url.split('/')[-2]
    dst = url.split('/')[-1]
    try:
        os.mkdir(DIR + folder)
    except FileExistsError as e:
        pass
    destination = DIR + folder + '/' + dst
    if os.path.isfile(destination):
        continue
    logger.debug('Downloading {0}'.format(url))
    get_large_file(url, destination)

for path, subdirs, files in os.walk(DIR):
    for name in files:
        if name.endswith('.zip'):
            dst = path + name
            logger.debug('Extracting {0}'.format(dst))
            subprocess.call('unzip',dst, '-d', path)
            subprocess.call('rm', dst)

for path, subdirs, files in os.walk(DIR):
    for name in files:
        if name.endswith('.txt'):
            dst = path + name
            logger.debug('Converting {0}'.format(dst))
            destination = dst.rstrip('.txt') + '.model'
            if os.path.isfile(destination):
                continue
            model = KeyedVectors.load_word2vec_format(dst)
            model.save(destination)