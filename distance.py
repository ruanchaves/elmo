import math
import numpy as np
from flair.data import Sentence
import sys
from sklearn.metrics.pairwise import cosine_similarity


def gensim_distance(sentences1, sentences2, model=None):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = [x.text for x in sent1.tokens]
        tokens2 = [x.text for x in sent2.tokens]

        # tokens1 = list(filter(lambda x: x in model.wv.vocab, tokens1))
        # tokens2 = list(filter(lambda x: x in model.wv.vocab, tokens2))

        tokens1 = [x if x in model.wv.vocab else 'unk' for x in tokens1]
        tokens2 = [x if x in model.wv.vocab else 'unk' for x in tokens2]

        embeddings_map1 = [ model.wv[x] for x in tokens1 ]
        embeddings_map2 = [ model.wv[x] for x in tokens2 ]

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        embedding1 = np.sum(embeddings_map1, axis=0).reshape(1, -1)
        embedding2 = np.sum(embeddings_map2, axis=0).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims

# Function adapted from https://github.com/TharinduDR/Simple-Sentence-Similarity/tree/master/matrices/context_vectors
def flair_distance(sentences1, sentences2, model=None):

    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = [x.text for x in sent1.tokens]
        tokens2 = [x.text for x in sent2.tokens]

        flair_sent1 = sent1
        flair_sent2 = sent2

        model.embed(flair_sent1)
        model.embed(flair_sent2)

        embeddings_map1 = {}
        embeddings_map2 = {}

        for token in flair_sent1:
            embeddings_map1[token.text] = np.array(token.embedding.data.tolist())

        for token in flair_sent2:
            embeddings_map2[token.text] = np.array(token.embedding.data.tolist())

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        embedding1 = np.sum([embeddings_map1[token] for token in tokens1], axis=0).reshape(1, -1)
        embedding2 = np.sum([embeddings_map2[token] for token in tokens2], axis=0).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims       

def combined_distance(sentences1, sentences2, gensim_model=None, flair_model=None):
    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = [x.text for x in sent1.tokens]
        tokens2 = [x.text for x in sent2.tokens]

        tokens1 = list(filter(lambda x: x in gensim_model.wv.vocab, tokens1))
        tokens2 = list(filter(lambda x: x in gensim_model.wv.vocab, tokens2))

        embeddings_map1 = [ gensim_model.wv[x] for x in tokens1 ]
        embeddings_map2 = [ gensim_model.wv[x] for x in tokens2 ]

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        gensim_embedding1 = np.average(embeddings_map1, axis=0).reshape(1, -1)
        gensim_embedding2 = np.average(embeddings_map2, axis=0).reshape(1, -1)

        tokens1 = [x.text for x in sent1.tokens]
        tokens2 = [x.text for x in sent2.tokens]

        flair_sent1 = sent1
        flair_sent2 = sent2

        flair_model.model.embed(flair_sent1)
        flair_model.model.embed(flair_sent2)

        embeddings_map1 = {}
        embeddings_map2 = {}

        for token in flair_sent1:
            embeddings_map1[token.text] = np.array(token.embedding.data.tolist())

        for token in flair_sent2:
            embeddings_map2[token.text] = np.array(token.embedding.data.tolist())

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        flair_embedding1 = np.average([embeddings_map1[token] for token in tokens1], axis=0).reshape(1, -1)
        flair_embedding2 = np.average([embeddings_map2[token] for token in tokens2], axis=0).reshape(1, -1)

        embedding1 = np.concatenate((gensim_embedding1, flair_embedding1), axis=1)
        embedding2 = np.concatenate((gensim_embedding2, flair_embedding2), axis=1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims       

if __name__ == '__main__':
    pass