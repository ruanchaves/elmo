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

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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
        self.attributes = self.generate(fname)
        self.vectors = self.enum(**self.attributes)

    def padding(self, lst, max_len=None, key='-PAD-'):
        for idx, item in enumerate(lst):
            tmp = item[::]
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

            get_flat = lambda x: sorted(list(set([ z for y in x for z in y ])))
            tag_list = get_flat(sentence_tags)
            tag_enum = { v: i + 1 for i, v in enumerate(tag_list) }
            tag_enum['-PAD-'] = 0

            word_list = get_flat(sentences)
            word_enum = { v.lower() : i + 2 for i, v in enumerate(word_list) }
            word_enum['-PAD-'] = 0
            word_enum['-OOV-'] = 1
            word_enum_reverse = { v : i for i, v in word_enum.items() }

            return {
                "sentences": sentences,
                "sentence_tags": sentence_tags,
                "word_enum": word_enum,
                "word_enum_reverse": word_enum_reverse,
                "tag_enum": tag_enum,
                "len_array": len_array
            }

    def enum(self, sentences=None, sentence_tags=None, word_enum=None, word_enum_reverse=None, tag_enum=None, len_array=None):
        num_sentences = []
        for sent in sentences:
            num_sent = []
            for word in sent:
                try:
                    num_sent.append(word_enum[word.lower()])
                except:
                    num_sent.append(word_enum['-OOV-'])
            num_sentences.append(num_sent)

        num_tags = []
        for tags in sentence_tags:
            num_t = []
            for t in tags:
                num_t.append(tag_enum[t])
            num_tags.append(num_t)
    
        max_len = len(max(num_sentences, key=len))
        num_sentences = pad_sequences(num_sentences, maxlen=max_len, padding='post')
        num_tags = pad_sequences(num_tags, maxlen=max_len, padding='post')

        return {
            "sentences": num_sentences,
            "sentence_tags": num_tags,
            "max_len": max_len
        }
                
def train_test_split_custom(data, attr):
    split_pos = data.attributes['len_array'][0]
    train = data.vectors[attr][0:split_pos] 
    test = data.vectors[attr][split_pos:len(data.attributes[attr])]    
    return train, test

source = "http://nilc.icmc.usp.br/macmorpho/macmorpho-v3.tgz"
fnames = ['macmorpho-dev.txt', 'macmorpho-test.txt', 'macmorpho-train.txt']
download(source, fnames)


data = Document(["macmorpho-train.txt", 'macmorpho-test.txt'])

X_train, X_test = train_test_split_custom(data, "sentences")
y_train, y_test = train_test_split_custom(data, "sentence_tags")

test_tensor = tf.convert_to_tensor(X_train[0:5])

# embedding = ELMoEmbeddings('pt')
session = tf.Session()

def ElmoEmbedding(x):
    x_arr = session.run(x)
    # embeddings = []
    # for sent in x_arr:
    #     tmp = ' '.join([ data.attributes['word_enum_reverse'][i] for i in sent ])
    #     tmp_sent = Sentence(tmp)
    #     embedding.embed(tmp_sent)
    #     tmp_embedding = [ token.embedding.data.tolist() for token in tmp_sent.tokens ]
    #     embeddings.append(tmp_embedding)
    # return tf.convert_to_tensor(x_arr)
    return tf.convert_to_tensor(x_arr)

max_len = data.vectors['max_len']
model = Sequential([
    InputLayer(input_shape=(max_len, )),
    Lambda(ElmoEmbedding, output_shape=(5, max_len, 3072))
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

labels = [ np.random.randint(2) for x in range(5) ]

model.fit(X_train[0:5], labels, epochs=50, verbose=0)

# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))