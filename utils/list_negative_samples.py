import numpy as np
import sys
import pdb
from time import time
import scipy.sparse as sp


class ListNegativeSamples(object):
    def __init__(self, train_matrix_item_seq, num_negatives, params, loss_criterion='ce'): ## treat user as list and item as item
        # param assignment
        self.train_matrix_item_seq        = train_matrix_item_seq
        self.num_negatives                = num_negatives
        self.loss_criterion               = loss_criterion

        self.list_input                   = np.arange(1,params.num_list)
        self.seq, self.seq_pos            = self.get_seq_and_seq_pos(train_matrix_item_seq)
        self.num_item = params.num_item
        self.params   = params
        #pdb.set_trace()

    def get_seq_and_seq_pos(self, seq_mat):
        seq_in_mat       = np.roll(seq_mat, 1, axis=1)
        seq_in_mat[:,0]  = 0
        seq_out_mat      = seq_mat.copy()
        row = np.arange(0,len(seq_mat))
        seq_out_mat[row,(seq_mat!=0).argmax(axis=1)] = 0
        return seq_in_mat[1:len(seq_mat)], seq_out_mat[1:len(seq_mat)]

    def generate_negative_seq_mat(self, seq_pos):
        #pdb.set_trace()
        num_row, num_col = seq_pos.shape
        seq_neg = np.random.choice(self.num_item-1, num_row * num_col).reshape(num_row, num_col) + 1 # -1 and +1 is added to avoid padded value (that is 0)
        seq_neg = seq_neg * (seq_pos != 0)

        return seq_neg

    # call this function from outside to generate instances at each epochs
    def generate_instances(self,):
        self.seq_neg = self.generate_negative_seq_mat(self.seq_pos)
        return self.list_input, self.seq, self.seq_pos, self.seq_neg
