import json
import yaml
import re
from collections import defaultdict
from collections import Counter
import spacy
import psycopg2

if __name__ == '__main__':

    stats = []    
    settings = {}
    with open("settings.yaml", 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    
    wiki_path = settings['wikipedia']['path']
    wiki_dump = settings['wikipedia']['dump']

    files_path = settings['ASSIN']['path']
    files = settings['ASSIN']['files']['ptbr']
    files2 = settings['ASSIN']['files']['ptpt']

    # nlp = spacy.load('pt_core_news_sm')
    # wiki = wiki_path + wiki_dump
    # with open(wiki, 'r') as f:
    #     for line in f:
    #         wiki_text = line
    #         doc = nlp(wiki_text, disable=['parser', 'tagger', 'ner'])
    #         words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    #         for w in words:
    #             with open(wiki_path + "wfreq.txt","a+") as f:
    #                 print(w, file=f)

    all_files = [files, files2]

    tokens = []
    for item in all_files:
        for fname in item.values():
            with open(files_path + fname,'r') as f:
                tokens.extend(json.load(f))
    
    flat_tokens = []
    for t in tokens:
        flat_tokens.extend(t['h'])
        flat_tokens.extend(t['t'])

    flat_tokens = list(set(flat_tokens))
    freqs = defaultdict(int)

    wiki = wiki_path + 'wfreq.txt'
    with open(wiki, 'r') as f:
        data = f.read().split('\n')
        with open(wiki_path + 'freqdict.json', 'w+') as f:
            json.dump(Counter(data).most_common(None), f)