import json
import subprocess
import shutil
from urllib.request import urlopen
import sys
import os
import subprocess
from loguru import logger

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

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

for path, subdirs, files in os.walk(NILC_DIR):
    for name in files:
        if name.endswith('.zip'):
            dst = path + name
            logger.debug('Extracting {0}'.format(dst))
            subprocess.call('unzip',dst, '-d', path)

for path, subdirs, files in os.walk(NILC_DIR):
    for name in files:
        if name.endswith('.txt'):
            dst = path + name
            logger.debug('Converting {0}'.format(dst))
            destination = dst.rstrip('.txt') + '.model'
            if os.path.isfile(destination):
                continue
            model = KeyedVectors.load_word2vec_format(dst)
            model.save(destination)