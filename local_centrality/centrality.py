from gensim.models import KeyedVectors
import json
from loguru import logger
import yaml
import sys
import os
import random
import datetime
import numpy as np
import copy
from gensim.similarities.index import AnnoyIndexer
import collections

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def save_results(fname, results):
    with open(fname,'w+') as f:
        json.dump(results, f)

def get_NILC(EMBEDDINGS_DIR):
    for path, subdirs, files in os.walk(EMBEDDINGS_DIR):
            for name in files:
                dst = path + '/' + name
                if name.endswith('.model'):
                    yield dst

def breadth_first_search()

def get_centrality(key=None, model=None, indexer=None, threshold=0.8):
    sims = model.most_similar(key, indexer=indexer, topn=100)
    triangles = 0
    indegree_centrality = 0
    sims = [x for x in sims if x[1] > threshold]
    sims = [x for x in sims if x[0] != key]
    average_distance = np.sum([1 - x[1] for x in sims])
    neighborhood = []
    for neighbor in sims:
        neighbor_key = neighbor[0]
        neighborhood.append(neighbor_key)
        sims2 = model.most_similar(neighbor_key, indexer=indexer, topn=100)
        sims2 = [x for x in sims2 if x[1] > threshold]
        sims2 = [x for x in sims2 if x[0] != key]
        sims2 = [x for x in sims2 if x[0] != neighbor_key]
        a = [ x[0] for x in sims ]
        b = [ x[0] for x in sims2 ]
        c = [ x for x in a if x in b ]
        triangles += len(c)
    coef = ( len(sims) * ( len(sims) - 1 ) )
    if coef:
        local_clustering_coefficient = triangles / coef
    else:
        local_clustering_coefficient = 0
    
    return {
    "local_clustering_coefficient": local_clustering_coefficient,
    "distance_sum": average_distance,
    "neighborhood": neighborhood
    "n_neighbors": len(neighborhood)
    }

def get_centrality_measures(key=None, dst=None, threshold=0.8, experiments=15):
    model = KeyedVectors.load(dst)
    indexer = AnnoyIndexer(model, num_trees=20)

    visited, queue = set(), collections.deque()
    visited.add(key)
    queue.append((key,0))
    depth = 0
    LIMIT = 2
    while queue:
        node, node_depth = queue.popleft()
        if node_depth > depth:
            depth += 1

        node_data = get_centrality(key=node, model=model, indexer=indexer, threshold=threshold)
        
        local_result = { 
            "local_clustering_coefficient": node_data["local_clustering_coefficient"],
            "distance_sum": node_data["distance_sum"],
            "n_neighbors": node_data["n_neighbors"]
            "node": node
            }

        result.append(local_result)

        if node_depth > LIMIT:
            continue
        
        for neighbor in node_data['neighborhood']:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    df = pd.DataFrame(result)
    global_clustering_coefficient = df["local_clustering_coefficient"].sum() / len(visited)
    average_short_distance = df["distance_sum"].sum() / df["n_neighbors"].sum()
    


if __name__ == '__main__':
    settings = {}
    tests = {}
    results = []

    os.chdir('..')
    settings = load_yaml("settings.yaml")

    LOGS_PATH = settings['logs']['path']
    RESULTS_PATH = settings['results']['path']

    EMBEDDINGS_DIR = settings['NILC']['dir']

    models = get_NILC(EMBEDDINGS_DIR)

    logger.add(LOGS_PATH + "centrality_{time}.log")
    RESULTS_FILE = RESULTS_PATH + 'centrality-' + str(int(datetime.datetime.now().timestamp())) + '.json'

    for source in models:
        query = 'unk'
        for result in get_centrality_measures(query, source, threshold=0.5):
            result["model"] = source
            results.append(result)
            logger.debug(result)
            save_results(RESULTS_FILE, results)