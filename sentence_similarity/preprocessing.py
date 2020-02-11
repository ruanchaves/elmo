import os
import copy
import json
import xml.etree.ElementTree as ET
from sys import stdout
import argparse
import re
import nltk
import yaml 
from utils import load_yaml

class JSONReader(object):
    
    def __init__(self, path, xml_files):
        self.path = path
        self.xml_files = xml_files

    def walk(self):
        file_data = []
        for name in self.xml_files:
            full_path = os.path.join(self.path, name)
            file_data.append(
                {
                    "path": full_path,
                    "root": self.path,
                    "name": name
                }
            )
        return file_data

    def get_json(self, fnames):
        entailment_dict = {
            "None": 0,
            "Entailment": 1,
            "Paraphrase": 2
        }
        for xml_file in fnames:
            tree = ET.parse(xml_file['path'])
            root = tree.getroot()
            df = []
            for idx, pair in enumerate(root.iter('pair')):
                if pair.get('id').endswith('-rev'):
                    continue
                sentence1 = pair.find('t').text
                sentence2 = pair.find('h').text

                row = { "id_": idx,
                    "t": sentence1,
                    "h": sentence2,
                    "similarity": pair.get('similarity'),
                    "entailment": entailment_dict[pair.get('entailment')]
                    }
                df.append(row)
            yield {
                "df": df,
                "root": xml_file['root'],
                "path": xml_file['path'],
                "name": xml_file['name']
            }

class Tokenizer(object):

    def __init__(self, path, xml_files, nltk_tokenizer = 'tokenizers/punkt/portuguese.pickle'):
        self.path = path
        self.xml_files = xml_files
        try:
            self.nltk_sent_tokenizer = nltk.data.load(nltk_tokenizer)
        except:
            nltk.download('punkt')
            self.nltk_sent_tokenizer = nltk.data.load(nltk_tokenizer)

    def NILC_tokenize(self, sentence):
        # Punctuation list
        punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')
        # ##### #
        # Regex #
        # ##### #
        re_remove_brackets = re.compile(r'\{.*\}')
        re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
        re_transform_numbers = re.compile(r'\d', re.UNICODE)
        re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
        re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
        # Different quotes are used.
        re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
        re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
        re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
        re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
        re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
        re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
        re_tree_dots = re.compile(u'…', re.UNICODE)
        # Differents punctuation patterns are used.
        re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                            (punctuations, punctuations), re.UNICODE)
        re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                                (punctuations, punctuations), re.UNICODE)
        re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
        re_changehyphen = re.compile(u'–')
        re_doublequotes_1 = re.compile(r'(\"\")')
        re_doublequotes_2 = re.compile(r'(\'\')')
        re_trim = re.compile(r' +', re.UNICODE)
        tokens = []
        for text in self.nltk_sent_tokenizer.tokenize(sentence):
            text = text.lower()
            text = text.replace('\xa0', ' ')
            text = re_tree_dots.sub('...', text)
            text = re.sub('\.\.\.', '', text)
            text = re_remove_brackets.sub('', text)
            text = re_changehyphen.sub('-', text)
            text = re_remove_html.sub(' ', text)
            text = re_transform_numbers.sub('0', text)
            text = re_transform_url.sub('URL', text)
            text = re_transform_emails.sub('EMAIL', text)
            text = re_quotes_1.sub(r'\1"', text)
            text = re_quotes_2.sub(r'"\1', text)
            text = re_quotes_3.sub('"', text)
            text = re.sub('"', '', text)
            text = re_dots.sub('.', text)
            text = re_punctuation.sub(r'\1', text)
            text = re_hiphen.sub(' - ', text)
            text = re_punkts.sub(r'\1 \2 \3', text)
            text = re_punkts_b.sub(r'\1 \2 \3', text)
            text = re_punkts_c.sub(r'\1 \2', text)
            text = re_doublequotes_1.sub('\"', text)
            text = re_doublequotes_2.sub('\'', text)
            text = re_trim.sub(' ', text)
            tokens.extend(text.strip().split(' '))
        return tokens

    def tokenize(self):
        reader = JSONReader(self.path, self.xml_files)
        fnames = reader.walk()
        sentence_dict = {}
        print(fnames)
        root = None
        with open('stopwords.json','r') as f:
            stopwords = json.load(f)
        for item in reader.get_json(fnames):
            root = item['root']
            file_df = []
            target = os.path.join(item['root'], item['name'].rstrip('xml') + 'json')
            for entry in item['df']:
                file_df.append(entry)
                t_array = self.NILC_tokenize(entry['t'])
                h_array = self.NILC_tokenize(entry['h'])
                
                t_array = [ x for x in t_array if len(x) > 1 ]
                h_array = [ x for x in h_array if len(x) > 1 ]

                t_array = [ x for x in t_array if x not in stopwords ]
                h_array = [ x for x in h_array if x not in stopwords ]  

                sentence_dict[entry['t']] = t_array
                sentence_dict[entry['h']] = h_array
            with open(target, 'w+') as f:
                json.dump(file_df, f)
        else:
            target = os.path.join(root, 'dictionary.json')
            with open(target,'w+') as f:
                json.dump(sentence_dict, f)
            
if __name__ == '__main__':
    settings = load_yaml("settings.yaml")
    path = settings['preprocessing']['path']
    xml_files = settings['preprocessing']['xml_files']
    tokenizer = Tokenizer(path, xml_files)
    tokenizer.tokenize()