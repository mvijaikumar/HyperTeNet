import pickle
import itertools
import pdb
from time import time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from data.dataset import Dataset
from data.tenet_dataset import TenetDataset
from utils import utils
import torch


class EmbedDataset(TenetDataset):
    def __init__(self, args):
        TenetDataset.__init__(self, args)
        self.user_edge_index, self.list_edge_index, self.item_edge_index = utils.load_pickle(args.path + '/' + args.embed_type + '/' + str(args.knn_k) + '/' + args.dataset + '.user_list_item_knn.pkl')
