import pickle
import itertools
from time import time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from utils import utils


class Dataset(object):
    def __init__(self,args):
        self.path                    = args.path + args.dataset
        self.method                  = args.method
        self.max_item_seq_length     = args.max_item_seq_length
        self.load_embedding_flag     = True if args.load_embedding_flag == 1 else False

        # load user_lists_dct, list_items_train, valid, test ========================
        self.user_lists_dct          = utils.load_pickle(self.path+'.user_lists.ided.pkl')
        #(self.list_items_dct_no_valid,self.list_items_dct,self.validNegativesDict,
        # self.testNegativesDict)     = self.get_pickle_train_valid_test(self.path+'.list_items.train_valid_test.pkl')
        (self.list_items_dct,self.list_items_wv_dct,self.validNegativesDict,
         self.testNegativesDict)     = self.get_pickle_train_valid_test(self.path+'.list_items.train_valid_test.pkl')

        ## change ===================
        ##self.validNegativesDict      = self.testNegativesDict ##to-be commented out
        self.list_items_dct_train    = self.list_items_dct
        self.list_items_dct          = self.list_items_wv_dct
        # ===========================

        self.list_user_dct           = self.get_list_user_dct(self.user_lists_dct) #user-embedding can be obtained using this(many-to-one)

        self.num_user,self.num_list,self.num_item = self.get_user_list_item_count(self.user_lists_dct, self.list_items_dct)
        self.list_user_vec           = self.get_list_user_vec(self.list_user_dct,self.num_list) #user-embedding can be obtained using this(many-to-one)

        # train valid test array-lists and sequence matrix ==========================
        self.trainArrTriplets        = self.get_arraylist_from_train_dict(self.list_items_dct)
        self.validArrDubles          = self.get_arraylist_from_valid_dict(self.validNegativesDict)
        self.testArrDubles           = self.get_arraylist_from_valid_dict(self.testNegativesDict)
        #===

        self.train_matrix            = self.get_train_matrix_sp(self.list_items_dct,self.num_list,self.num_item)
        self.train_matrix_item_seq   = self.get_dct_mat_seq_remove_test(dct=self.list_items_dct, num_row=self.num_list, num_col=self.max_item_seq_length+1, padding_value=0) ##last_index :] for all, :-1] for remove test item, :-2] for removing test and valid
        self.train_matrix_item_seq_for_test = self.get_dct_mat_seq_for_test(dct=self.list_items_dct, num_row=self.num_list, num_col=self.max_item_seq_length, padding_value=0) ##last_index :] for all, :-1] for remove test item, :-2] for removing test and valid

    def get_pickle_train_valid_test(self,fname):
        return utils.load_pickle(fname)

    def get_list_user_dct(self, user_lists_dct):
        list_user_dct = dict()
        for user in user_lists_dct:
            for l in user_lists_dct[user]:
                assert l not in list_user_dct, 'lists are not uniqe. data-extraction some part is wrong.'
                list_user_dct[l] = user
        return list_user_dct


    def get_list_user_vec(self, list_user_dct, num_list):
        list_user_arr = np.zeros(num_list,dtype=np.int)
        for l in list_user_dct:
            list_user_arr[l] = list_user_dct[l]
        return list_user_arr

    def get_user_list_item_count(self, user_lists_dct, list_items_dct):
        num_user   = max(user_lists_dct.keys())+1
        num_list   = max(list_items_dct.keys())+1
        num_list2  = max(set(itertools.chain.from_iterable(self.user_lists_dct.values())))+1

        # assert statements for list matching across user_lists and list_items dct (thru train_dct)
        #''' ## remove comments
        assert num_list == num_list2, 'max list indices from list_items_dct and user_lists_dct should be same.'
        assert len(list_items_dct.keys()) == len(set(itertools.chain.from_iterable(self.user_lists_dct.values()))),'number of lists from both the list_items_dct and user_lists_dct should be same.'
        #'''
        num_item   = max(set(itertools.chain.from_iterable(self.list_items_dct.values())))+1
        return num_user, num_list, num_item

    def get_arraylist_from_train_dict(self, train_dct):
        list_input, item_input, rating = [],[],[]
        for l in train_dct:
            items = train_dct[l]
            for item in items:
                list_input.append(l)
                item_input.append(item)
                rating.append(1.0)
        return np.array(list_input), np.array(item_input), np.array(rating,dtype=np.float16)

    def get_arraylist_from_valid_dict(self, valid_dct):
        list_input, item_input = [],[]
        for key in valid_dct:
            l, item = key
            list_input.append(l)
            item_input.append(item)
        return np.array(list_input), np.array(item_input)

    def get_dct_mat_seq(self, dct, num_row, num_col, padding_value):
        mat = np.full((num_row,num_col),padding_value,dtype=int)
        for lst in dct:
            items_arr = np.array(dct[lst])
            leng      = len(items_arr)
            if leng >= num_col:
                mat[lst,:]      = items_arr[-num_col:]
            else:
                mat[lst,-leng:] = items_arr
        return mat

    def get_dct_mat_seq_remove_test(self, dct, num_row, num_col, padding_value):
        mat = np.full((num_row,num_col-1),padding_value,dtype=int)
        for lst in dct:
            items_arr = np.array(dct[lst])
            leng      = len(items_arr)
            if leng >= num_col:
                mat[lst,:]      = items_arr[-num_col:-1]
            else:
                mat[lst,-leng+1:] = items_arr[:-1]
        return mat

    def get_dct_mat_seq_for_test(self, dct, num_row, num_col, padding_value):
        mat = np.full((num_row,num_col),padding_value,dtype=int)
        for lst in dct:
            items_arr = np.array(dct[lst])
            leng      = len(items_arr)
            if leng >= num_col:
                mat[lst,:]      = items_arr[-num_col:]
            else:
                mat[lst,-leng:] = items_arr[:]
        return mat

    def get_train_matrix_sp(self, dct, num_row, num_col):
        mat     = sp.dok_matrix((num_row, num_col), dtype=np.float16) ##float32
        for key in dct:
            for val in dct[key]:
                mat[key,val] = 1.0
        return mat.tolil()
