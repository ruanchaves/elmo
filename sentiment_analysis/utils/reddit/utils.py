import os
import yaml
import requests
import re
import shutil
from bs4 import BeautifulSoup
import json

from urllib.request import urlopen

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def get_filenames(url):
    text = requests.get(url).text
    soup = BeautifulSoup(text, 'lxml')
    result = []
    for link in soup.findAll('a', attrs={'href': re.compile("^.*\.(bz2|xz|zst)$")}):
        result.append(url + link.get('href').lstrip('./'))
    return [ v for i,v in enumerate(result) if result.index(v) == i ]

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

    