import requests
import tarfile 
import io
import os
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Lambda, Flatten
from keras.optimizers import Adam

from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence

import tensorflow as tf

from keras import backend as K
from keras.layers import Layer

import copy
import h5py
database = 'elmo.hdf5'

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

def download(source, files):
    if all([x in os.listdir() for x in files]):
        return True
    try:
        content = requests.get(source).content
        file_like_object = io.BytesIO(content)
        tar = tarfile.open(fileobj=file_like_object)
        for member in tar.getmembers():
            name = member.name
            print(name)
            tar.extract(member)
        return True
    except Exception as e:
        print(e)
        return False

class Document(object):

    def __init__(self, fname=None):
        self.elmo = ELMoEmbeddings('pt')
        self.attributes = self.generate(fname)
        self.embeddings = None

    def padding(self, lst, max_len=None, key='-PAD-'):
        for idx, item in enumerate(lst):
            tmp = list(item[::])
            while len(tmp) < max_len:
                tmp.append(key)
            lst[idx] = tmp
        return lst

    def generate(self, item):

        all_content = []
        len_array = []
        for fname in item:
            with open(fname, 'r') as f:
                content = f.read().split('\n')
                content = [ x.split(' ') for x in content ]
                content = [ [ tuple(y.split('_')) for y in x ] for x in content ]
                check = [all([len(y) == 2 for y in x]) for x in content]
                content = [ v for i,v in enumerate(content) if check[i] == True ]
            len_array.append(len(content))
            all_content.extend(content)

        sentences = []
        sentence_tags = []
        for sent in all_content:
            txt, tags = zip(*sent)
            sentences.append(np.array(txt))
            sentence_tags.append(np.array(tags))

        max_len = len(max(sentences, key=len))
        sentences = self.padding(sentences,max_len=max_len)
        sentence_tags = self.padding(sentence_tags, max_len=max_len)

        enum_array = copy.deepcopy(sentences)
        for idx, item in enumerate(enum_array):
            for idx2, item2 in enumerate(item):
                enum_array[idx][idx2] = idx*len(item) + idx2

        tag_list = sorted(list(set(np.array(sentence_tags).flatten().tolist())))
        tag_list.remove('-PAD-')
        tag_list.insert(0,'-PAD-')
        tag_dict = { v:i for i,v in enumerate(tag_list) }
        tag_array = copy.deepcopy(sentence_tags)
        tag_len = len(tag_list)

        for idx, item in enumerate(sentence_tags):
            for idx2, item2 in enumerate(item):
                arr = np.zeros(tag_len)
                arr[tag_dict[item2]] = 1.0
                tag_array[idx][idx2] = arr

        vocab_size = enum_array[-1][-1] + 1
        max_sequence_length = len(enum_array[-1])

        return {
            "sentences": sentences,
            "sentence_tags": sentence_tags,
            "len_array": len_array,
            "enum_array": enum_array,
            "tag_array": tag_array,
            "tag_len": tag_len,
            "vocab_size": vocab_size,
            "max_sequence_length": max_sequence_length
        }
        
    def get_elmo(self, key='-PAD-', filename=None, vocab_size=None):
        with h5py.File(database,'w') as f:
            f.create_dataset("data", (vocab_size, 3072), compression='gzip')
        sentences = self.attributes['sentences']
        enum_array = self.attributes['enum_array']
        embedding_dict = {}
        for idx, item in enumerate(enum_array):
            tmp = ' '.join(sentences[idx])
            sent = Sentence(tmp)
            self.elmo.embed(sent)
            for idx2, item2 in enumerate(item):
                if sent[idx2].text == '-PAD-':
                    continue
                else:
                    embedding_array = np.array(sent[idx2].embedding.data.tolist())
                    with h5py.File(database, 'a') as f:
                        f['data'][item2][:] = embedding_array
        # embedding_dict = sorted(embedding_dict.items(), key=lambda x: x[0])
        # embedding_dict = np.array([ x[1] for x in embedding_dict ])
        # if save:
        #     np.save(filename, embedding_dict, allow_pickle=False)
        # return embedding_dict


def train_test_split_custom(data, attr):
    split_array = [0] + data.attributes['len_array']
    split_array = np.cumsum(split_array)
    split_array = list(zip(split_array, split_array[1:]))
    return ( np.array(data.attributes[attr][tup[0]:tup[1]]) for tup in split_array )

source = "http://nilc.icmc.usp.br/macmorpho/macmorpho-v3.tgz"
fnames = ['macmorpho-dev.txt', 'macmorpho-test.txt', 'macmorpho-train.txt']
download(source, fnames)


data = Document(["macmorpho-train.txt", "macmorpho-test.txt", "macmorpho-dev.txt"])

vocab_size = data.attributes['vocab_size']
data.get_elmo(filename=database, vocab_size=vocab_size)

# X_train, X_test, X_val = train_test_split_custom(data, "enum_array")
# y_train, y_test, y_val = train_test_split_custom(data, "tag_array")

# weights = data.embeddings
# vocab_size = data.attributes['vocab_size']
# max_len = data.attributes['max_sequence_length']
# tag_len = data.attributes['tag_len']

# model = Sequential([
#     Embedding(vocab_size, 3072, weights=[weights], input_length=max_len, trainable=False),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     TimeDistributed(Dense(tag_len)),
#     Activation('softmax')
# ])

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(0.001),
#               metrics=['accuracy', ignore_class_accuracy(0)])
 
# model.summary()

# model.fit(X_train, y_train, batch_size=128, epochs=40, validation_data=(X_val, y_val) )

# scores = model.evaluate(X_test, y_test)
# print(f"{model.metrics_names[1]}: {scores[1] * 100}")