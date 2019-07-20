# -*- coding: utf-8 -*-

"""
Common structures and functions used by other scripts.

Adapted from ASSIN - https://github.com/erickrf/assin/blob/master/commons.py
"""

from xml.etree import cElementTree as ET
import os
import json
import pickle

str_to_entailment = {'none': 0,
                     'entailment': 1,
                     'paraphrase': 2}
entailment_to_str = {v: k for k, v in str_to_entailment.items()}

def pickle_to_csv(filename):
    pairs = []
    with open(filename, 'rb') as f:
        pair_list = pickle.load(f)
    for idx, pickle_pair in enumerate(pair_list):
        tokens_t1 = pickle_pair['tokens_t1']
        tokens_t2 = pickle_pair['tokens_t2']
        result = pickle_pair['result']
        pair = {
            't': tokens_t1, # text
            'h': tokens_t2, # hypothesis
            'id_': idx,
            'entailment': -1,
            'similarity': result
        }
        pairs.append(pair)
    return pairs


def xml_to_csv(filename):
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()

    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        id_ = attribs['id']

        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()

            try:
                ent_value = str_to_entailment[ent_string]
            except ValueError:
                msg = 'Unexpected value for attribute "entailment" at pair {}: {}'
                raise ValueError(msg.format(id_, ent_string))

        else:
            ent_value = None

        if 'similarity' in attribs:
            similarity = float(attribs['similarity'])
        else:
            similarity = None

        if similarity is None and ent_value is None:
            msg = 'Missing both entailment and similarity values for pair {}'.format(id_)
            raise ValueError(msg)

        pair = {
            't': t, # text
            'h': h, # hypothesis
            'id_': id_,
            'entailment': ent_value,
            'similarity': similarity
        }
        pairs.append(pair)

    return pairs

if __name__ == '__main__':

    os.chdir('..')

    DIR = './datasets/ASSIN/'
    for item in os.listdir(DIR):
        if 'xml' in item and '._' not in item:
            data = xml_to_csv(DIR + item)
            fname = item.split('.')[0] + '-standard.json'
            with open(DIR + fname, 'w+') as f:
                json.dump(data, f)

    PICKLE_DIR = '../portuguese_word_embeddings/sentence_similarity/data/'
    for item in os.listdir(PICKLE_DIR):
        if 'pkl' in item:
            data = pickle_to_csv(PICKLE_DIR + item)
            fname = item.split('.')[0] + '-hartmann.json'
            with open(DIR + fname, 'w+') as f:
                json.dump(data, f)
