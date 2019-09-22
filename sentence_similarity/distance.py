import math
import numpy as np
from flair.data import Sentence
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import copy
import json
from utils import sentence_to_hash
from flair.data import Sentence
# Major parts of this code have been adapted from https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/matrices/context_vectors .

def remove_first_principal_component(X):
    Y = np.array([ w.flatten() for w in X])
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(Y)
    pc = svd.components_
    XX = Y - Y.dot(pc.transpose()) * pc
    return XX

def safe_division(a, freqs, token, total_freq):
    try:
        p_w = freqs[token] / total_freq
        return a / (a + p_w)
    except ZeroDivisionError as e:
        return a / a

def cosine_distance(
    sentences1, 
    sentences2, 
    gensim_model=None, 
    flair_model=None,
    bert_model=None, 
    gensim_sif=False, 
    flair_sif=False, 
    freqs={}, 
    total_freq=1.0, 
    a=0.001, 
    unk=False,
    dictionary=None):
    
    sims = {}
    embeddings = []
    embeddings_pos = []
    equivalence = False
    for sent_index, (sent1, sent2) in enumerate(zip(sentences1, sentences2)):
        tokens1 = [x.text for x in Sentence(sent1).tokens]
        tokens2 = [x.text for x in Sentence(sent2).tokens]

        if bert_model:

            embedding1 = bert_model.encode([sent1])[0]
            embedding2 = bert_model.encode([sent2])[0]    

        if gensim_model:

            if equivalence == True:
                gensim_tokens1 = dictionary[sent1]
                gensim_tokens2 = dictionary[sent2]
            else:
                gensim_tokens1 = copy.deepcopy(tokens1)
                gensim_tokens2 = copy.deepcopy(tokens2)

            # Right procedure
            if unk == False:
                known_tokens1 = list(filter(lambda x: x in gensim_model.wv.vocab, gensim_tokens1))
                known_tokens2 = list(filter(lambda x: x in gensim_model.wv.vocab, gensim_tokens2))
            # Wrong procedure for FastText
            elif unk == True:
                known_tokens1 = [x if x in gensim_model.wv.vocab else 'unk' for x in gensim_tokens1]
                known_tokens2 = [x if x in gensim_model.wv.vocab else 'unk' for x in gensim_tokens2]

            if len(known_tokens1) == 0 or len(known_tokens2) == 0:
                sims[sent_index] = 0
                continue

            if gensim_sif:              
                weights1 = [ safe_division(a, freqs, token, total_freq) for token in known_tokens1]
                weights2 = [ safe_division(a, freqs, token, total_freq) for token in known_tokens2]  

            embeddings_map1 = [ gensim_model.wv[x] for x in known_tokens1 ]
            embeddings_map2 = [ gensim_model.wv[x] for x in known_tokens2 ]

            if gensim_sif:         
                gensim_embedding1 = np.average(embeddings_map1, axis=0, weights=weights1).reshape(1, -1)
                gensim_embedding2 = np.average(embeddings_map2, axis=0, weights=weights2).reshape(1, -1)
            else:
                gensim_embedding1 = np.sum(embeddings_map1, axis=0).reshape(1, -1)
                gensim_embedding2 = np.sum(embeddings_map2, axis=0).reshape(1, -1)
            
        if flair_model:

            if len(tokens1) == 0 or len(tokens2) == 0:
                sims[sent_index] = 0
                continue

            flair_sent1 = Sentence(sent1)
            flair_sent2 = Sentence(sent2)

            flair_model.embed(flair_sent1)
            flair_model.embed(flair_sent2)

            embeddings_map1 = {}
            embeddings_map2 = {}

            for token in flair_sent1:
                embeddings_map1[token.text] = np.array(token.embedding.data.tolist())

            for token in flair_sent2:
                embeddings_map2[token.text] = np.array(token.embedding.data.tolist())

            if flair_sif:
                weights1 = [ safe_division(a, freqs, token, total_freq) for token in tokens1]
                weights2 = [ safe_division(a, freqs, token, total_freq) for token in tokens2]
            else:
                weights1 = None
                weights2 = None

            flair_embedding1 = np.average([embeddings_map1[token] for token in tokens1], axis=0, weights=weights1).reshape(1, -1)
            flair_embedding2 = np.average([embeddings_map2[token] for token in tokens2], axis=0, weights=weights2).reshape(1, -1)

        if flair_model and gensim_model:
            embedding1 = np.concatenate((gensim_embedding1, flair_embedding1), axis=1)
            embedding2 = np.concatenate((gensim_embedding2, flair_embedding2), axis=1)
        elif flair_model and not gensim_model:
            embedding1 = flair_embedding1
            embedding2 = flair_embedding2
        elif not flair_model and gensim_model:
            embedding1 = gensim_embedding1
            embedding2 = gensim_embedding2

        embeddings.append(embedding1)
        embeddings.append(embedding2)
        embeddings_pos.append(sent_index)
    
    if flair_sif or gensim_sif:
        embeddings = remove_first_principal_component(np.array(embeddings))
    distances = [cosine_similarity(embeddings[idx * 2].reshape(1, -1),
                              embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings) / 2))]
    for idx,score in enumerate(distances):
        sims[embeddings_pos[idx]] = score
    sims = [x[1] for x in sorted(list(sims.items()), key=(lambda x: x[0]))]
    return sims                      

if __name__ == '__main__':
    pass