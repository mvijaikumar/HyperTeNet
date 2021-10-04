import pickle
import itertools
import pdb
from time import time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from data.dataset import Dataset
from utils import utils


class TenetDataset(Dataset):
    def __init__(self,args):
        Dataset.__init__(self,args)
        #pdb.set_trace()

        self.user_items_dct          = self.get_user_items_dict(self.user_lists_dct, self.list_items_dct)
        self.user_item_matrix_sp     = self.get_sparse_matrix_from_dict(self.user_items_dct, self.num_user, self.num_item)
        self.list_item_matrix_sp     = self.get_sparse_matrix_from_dict(self.list_items_dct, self.num_list, self.num_item)
        self.item_list_matrix_sp     = self.get_sparse_matrix_from_dict(self.list_items_dct, self.num_item, self.num_list, reverse=True)

        t1                   = time()
        self.user_user_comm_mat_sp   = self.mat_mult_sp(self.user_item_matrix_sp, self.user_item_matrix_sp.T)#.astype(bool).astype(int)#todok()
        self.item_item_comm_mat_sp   = self.mat_mult_sp(self.item_list_matrix_sp, self.item_list_matrix_sp.T)#.astype(bool).astype(int)
        self.list_list_comm_mat_sp   = self.mat_mult_sp(self.list_item_matrix_sp, self.list_item_matrix_sp.T)##.astype(bool).astype(int) ##real values
        #pdb.set_trace()

        # ==============================
        #self.list_item_train_seq     = self.get_dct_mat_seq(dct=self.list_items_dct, num_row=self.num_list, num_col=self.num_item, padding_value=0)

        ##self.train_matrix_item_seq   = self.get_dct_mat_seq_remove_test(dct=self.list_items_dct, num_row=self.num_list, num_col=self.max_item_seq_length+1, padding_value=0) ##last_index :] for all, :-1] for remove test item, :-2] for removing test and valid

        #pdb.set_trace()

        ##
        '''
        binarize = True
        if binarize == True:
            self.user_user_comm_mat_sp   = sp.csr_matrix(sp.csr_matrix((self.user_user_comm_mat_sp),dtype=bool),dtype=int)
            self.item_item_comm_mat_sp   = sp.csr_matrix(sp.csr_matrix((self.item_item_comm_mat_sp),dtype=bool),dtype=int)
            self.list_list_comm_mat_sp   = sp.csr_matrix(sp.csr_matrix((self.list_list_comm_mat_sp),dtype=bool),dtype=int)
            #self.user_user_comm_mat_sp   = self.binarize_sparse_matrix(self.user_user_comm_mat_sp)
            #self.item_item_comm_mat_sp   = self.binarize_sparse_matrix(self.item_item_comm_mat_sp)
            #self.list_list_comm_mat_sp   = self.binarize_sparse_matrix(self.list_list_comm_mat_sp)

        '''

        # adj ===========
        self.user_adj_mat = self.user_user_comm_mat_sp
        self.list_adj_mat = self.list_list_comm_mat_sp
        self.item_adj_mat = self.item_item_comm_mat_sp

    # ==========================================
    def mat_mult_sp(self, mat1, mat2):
        return mat1 * mat2 #.todtype(fset=int)

    def get_sparse_matrix_from_dict(self, dct, num_row, num_col, reverse=False):
        sp_mat = sp.lil_matrix((num_row,num_col))

        for key in dct:
            values = dct[key]
            for value in values:
                if reverse == False:
                    sp_mat[key,value] = 1
                else:
                    sp_mat[value,key] = 1
        return sp_mat

    def get_user_items_dict(self, user_lists_dct, list_items_dct):
        user_items_dct = defaultdict(set)
        for user in user_lists_dct:
            for lst in user_lists_dct[user]:
                #pdb.set_trace()
                # for binary. if freq is wanted, change this to list and count during the sp_matrix conv (first binarize)
                #print(set(list_items_dct[lst]))
                user_items_dct[user] = user_items_dct[user].union(set(list_items_dct[lst]))

        return user_items_dct

    def binarize_sparse_matrix(self, mat_sp):
        bin_mat_sp = sp.lil_matrix(mat_sp.shape, dtype=np.float32)
        for key in mat_sp.keys():
            bin_mat_sp[key[0],key[1]] = 1
        return bin_mat_sp

