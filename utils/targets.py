import yaml
import json
import os

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

if __name__ == '__main__':
    os.chdir('..')
    settings = load_yaml('settings.yaml')
    corpus = settings['ASSIN']['files']['total']
    path = settings['ASSIN']['path']
    EMBEDDINGS_DIR = settings['NILC']['dir']
    with open(path + corpus,'r') as f:
        tmp = f.read().split('\n')
    vocab = []
    for item in tmp:
        for word in item.split(' '):
            vocab.append(word)
    
    fnames = []
    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
            for name in files:
                dst = path + '/' + name
                if name.endswith('.txt'):
                    fnames.append(path + '/' + name)

    for item in fnames:
        words = []
        learn = []
        with open(item,'r') as f:
            for line in f:
                w = line.split(' ')[0]
                words.append(w)
        for v in vocab:
            if v not in words:
                learn.append(v)
        learn = '\n'.join(learn)
        name = item.split('/')[-1]
        folder = item.split('/')[-2] + '/'
        try:
            os.mkdir('./alacarte/' + folder)
        except:
            pass
        with open('./alacarte/' + folder + name,'w+') as f:
            print(learn, file=f)