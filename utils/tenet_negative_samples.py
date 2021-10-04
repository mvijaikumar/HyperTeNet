import numpy as np
import sys
import pdb
from time import time
import scipy.sparse as sp


class NegativeSamples(object):
    def __init__(self, sp_matrix, num_negatives, params, loss_criterion='ce'): ## treat user as list and item as item
        # param assignment
        self.sp_matrix        = sp_matrix
        self.num_negatives    = num_negatives
        self.loss_criterion   = loss_criterion
        self.num_rating       = sp_matrix.nnz
        self.num_item         = sp_matrix.shape[-1] ##

        ## new
        self.num_user         = params.num_user
        self.list_user_vec    = params.list_user_vec

        # positive part
        self.list_pos_arr,self.item_pos_arr,self.rating_pos_arr = self.get_positive_instances(sp_matrix)
        #pdb.set_trace()
        self.user_pos_arr   = self.list_user_vec[self.list_pos_arr]

        # negative part
        self.list_neg_arr   = np.repeat(self.list_pos_arr,self.num_negatives) ##negative samples could be different bw item and bundle
        self.rating_neg_arr = np.repeat([0],len(self.rating_pos_arr) * self.num_negatives)
        ##pdb.set_trace()

        # positive_and_negative part pre-generated to improve efficiency
        self.list_arr   = np.concatenate([self.list_pos_arr,self.list_neg_arr])
        self.rating_arr = np.concatenate([self.rating_pos_arr,self.rating_neg_arr])
        self.rating_arr = self.rating_arr.astype(np.float16)

    def get_positive_instances(self,mat):
        list_pos_arr,item_pos_arr,rating_pos_arr=(np.array([],dtype=np.int),np.array([],dtype=np.int),np.array([],dtype=np.int))
        pos_mat = mat.tocsc().tocoo()
        list_pos_arr,item_pos_arr = pos_mat.row,pos_mat.col
        rating_pos_arr = np.repeat([1],len(list_pos_arr))

        return list_pos_arr,item_pos_arr,rating_pos_arr

    def generate_negative_item_samples(self,):
        neg_item_arr = np.array([],dtype=np.int)
        if self.loss_criterion == 'pairwise':
            random_indices = np.random.choice(self.num_item-1, 1 * self.num_rating) + 1 ##to tackle 0-padding
        else:
            random_indices = np.random.choice(self.num_item-1, self.num_negatives * self.num_rating) + 1 ##to tackle 0-padding
        neg_item_arr = random_indices

        return neg_item_arr

    def generate_negative_user_samples(self,):
        neg_user_arr = np.array([],dtype=np.int)
        if self.loss_criterion == 'pairwise':
            random_indices = np.random.choice(self.num_user-1, 1 * self.num_rating) + 1 ##to tackle 0-padding
        else:
            random_indices = np.random.choice(self.num_user-1, self.num_negatives * self.num_rating) + 1 ##to tackle 0-padding
        neg_user_arr = random_indices

        return neg_user_arr

    # call this function from outside to generate instances at each epochs
    def generate_instances(self,):
        self.item_neg_arr = self.generate_negative_item_samples()
        self.item_arr     = np.concatenate([self.item_pos_arr,self.item_neg_arr])
        ##new
        self.user_neg_arr = self.generate_negative_user_samples()
        self.user_arr     = np.concatenate([self.user_pos_arr,self.user_neg_arr])
        return self.user_arr, self.list_arr, self.item_arr, self.rating_arr

    def generate_instances_bpr(self,):
        self.user_neg_arr = self.generate_negative_user_samples()
        self.item_neg_arr = self.generate_negative_item_samples()
        return self.user_pos_arr,self.user_neg_arr,self.list_pos_arr,self.item_pos_arr,self.item_neg_arr
