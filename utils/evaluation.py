import math, heapq, multiprocessing, sys, pdb
import numpy as np
from time import time

_posItemlst = None
_itemMatrix = None
_predMatrix = None
_k = None
_matShape = None


def evaluate_model(posItemlst, itemMatrix, predMatrix, k, num_thread):
    global _posItemlst
    global _itemMatrix
    global _predMatrix
    global _k
    global _matShape

    _posItemlst = posItemlst
    _itemMatrix = itemMatrix
    _predMatrix = predMatrix
    _k = k
    _matShape = itemMatrix.shape

    num_inst = _matShape[0]

    hits, ndcgs, maps = [], [], []
    if num_thread > 1:
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(num_inst))

        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        maps = [r[2] for r in res]
        return (hits, ndcgs, maps)

    # Single thread        
    for ind in range(num_inst):
        (hr, ndcg, mapval) = eval_one_rating(ind)
        hits.append(hr)
        ndcgs.append(ndcg)
        maps.append(mapval)
    # print ("hits and ndcgs: ",hits,ndcgs)
    return (hits, ndcgs, maps)


def eval_one_rating(ind):
    map_item_score = {}
    predictions = _predMatrix[ind]
    items = _itemMatrix[ind]
    gtItem = _posItemlst[ind]

    for i in range(_matShape[1]):  ## parallelaize by assigning array to array in dict
        item = items[i]
        map_item_score[item] = predictions[i]
    # Evaluate top rank list
    ranklist = heapq.nlargest(_k, map_item_score, key=map_item_score.get)
    # ranklist = heapq.nsmallest(_k, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    mapval = getMAP(ranklist, gtItem)
    # pdb.set_trace()
    return (hr, ndcg, mapval)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getMAP(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return 1.0 / (i + 1)
    return 0
