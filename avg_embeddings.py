# Adapted from https://github.com/TharinduDR/Simple-Sentence-Similarity/tree/master/matrices/context_vectors

import math
from collections import Counter

import numpy as np
from flair.data import Sentence
from sklearn.metrics.pairwise import cosine_similarity


def run_context_avg_benchmark(sentences1, sentences2, model=None):

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

        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)

        embedding1 = np.sum([embeddings_map1[token] for token in tokfreqs1], axis=0).reshape(1, -1)
        embedding2 = np.sum([embeddings_map2[token] for token in tokfreqs2], axis=0).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims

if __name__ == '__main__':
    pass