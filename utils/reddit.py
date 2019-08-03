import os
import yaml
import requests
import re
from bs4 import BeautifulSoup

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Text  
from sqlalchemy.dialects.postgresql import JSON, JSONB

import json  
import sqlalchemy  
db = sqlalchemy.create_engine(connection_string)  
engine = db.connect()  
meta = sqlalchemy.MetaData(engine)  

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def get_files(url):
    text = requests.get(url).text
    soup = BeautifulSoup(text, 'lxml')
    for link in soup.findAll('a', attrs={'href': re.compile("^.*\.(bz2|xz|zst)$")}):
        yield url + link.get('href').lstrip('./')

def get_large_file(url, file, length=16*1024):
    req = urlopen(url)
    with open(file, 'wb') as fp:
        shutil.copyfileobj(req, fp, length)

def decompress(fname):
    cmds = {
        'zst': 'unzstd', 
        'bz2': 'bzip2 -d',
        'xz': 'unxz'
    }
    cmd = cmds[fname.split('.')[-1]] + ' ' + fname
    cmd = cmd.split(' ')
    subprocess.call(cmd)
    return fname.split('.')[-2]



if __name__ == '__main__':
    os.chdir('..')
    settings = load_yaml('settings.yaml')
    url = settings['reddit']['pushshift']
    path = settings['reddit']['path']
    files = [ x for x in get_files(url) ][-1]

    sqlalchemy.Table("jsontable", meta,  
                Column('id', Integer),
                Column('name', Text),
                Column('email', Text),
                Column('doc', JSON))