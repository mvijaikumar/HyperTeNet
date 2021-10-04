import sys
import pdb
import random
import pickle
import numpy as np
import scipy.sparse as sp


class Parameters(object):
    def __init__(self, args, dataset):
        # Arguments =======================================================================
        # dataset and method ==============================================================
        self.args                     = args
        self.dataset_obj              = dataset
        self.method                   = args.method.lower()
        self.path                     = args.path
        self.dataset                  = args.dataset
        self.result_path              = args.res_path + args.dataset + '/' + args.method + '/'
        self.result_folder            = args.res_folder
        self.include_networks         = eval(args.include_networks)

        # algo-parameters =======================================================
        self.num_epochs               = args.num_epochs
        self.batch_size               = args.batch_size
        self.batch_size_seq           = args.batch_size_seq
        self.valid_batch_siz          = args.valid_batch_siz
        self.lr                       = args.lr
        self.optimizer                = args.optimizer
        self.loss                     = args.loss
        if self.method in ['bpr']:
            self.loss                 = 'bpr'
        self.initializer              = args.initializer
        self.stddev                   = args.stddev
        self.max_item_seq_length      = args.max_item_seq_length
        self.load_embedding_flag      = dataset.load_embedding_flag #indicates extra-information

        # hyper-parameters ======================================================
        self.num_factors              = args.num_factors
        self.num_layers               = args.num_layers ## testing
        self.num_negatives            = args.num_negatives
        self.num_negatives_seq        = args.num_negatives_seq
        self.reg_w                    = args.reg_w
        self.reg_b                    = args.reg_b
        self.reg_lambda               = args.reg_lambda
        self.margin                   = args.margin
        self.keep_prob                = args.keep_prob

        # gnn ==============================================================
        self.hid_units                = eval(args.hid_units)
        self.n_heads                  = eval(args.n_heads)
        self.gnn_keep_prob            = args.gnn_keep_prob
        self.net_keep_prob            = args.net_keep_prob
        self.d_k                      = args.d_k

        # valid test ============================================================
        self.at_k                     = args.at_k
        self.num_thread               = args.num_thread
        self.epoch_mod                = args.epoch_mod

        # Dataset ==============================================================
        # count ================================================================
        self.num_user                 = dataset.num_user
        self.num_list                 = dataset.num_list
        self.num_item                 = dataset.num_item
        self.num_train_instances      = len(dataset.trainArrTriplets[0])
        self.num_valid_instances      = len(dataset.validNegativesDict.keys())
        self.num_test_instances       = len(dataset.testNegativesDict.keys())
        self.num_nodes                = self.num_user + self.num_list + self.num_item

        # data-structures ======================================================
        self.user_lists_dct           = dataset.user_lists_dct
        self.list_items_dct           = dataset.list_items_dct # only training items
        self.list_items_dct_train     = dataset.list_items_dct_train ## make sure about proper train and validation data
        self.list_user_dct            = dataset.list_user_dct
        self.list_user_vec            = dataset.list_user_vec
        self.train_matrix             = dataset.train_matrix
        self.testNegativesDict        = dataset.testNegativesDict
        self.validNegativesDict       = dataset.validNegativesDict

        self.trainArrTriplets         = dataset.trainArrTriplets
        self.validArrDubles           = dataset.validArrDubles
        self.testArrDubles            = dataset.testArrDubles
        self.train_matrix_item_seq    = dataset.train_matrix_item_seq

        ##new
        self.train_matrix_item_seq_for_test    = dataset.train_matrix_item_seq_for_test

        # adj ===========
        if self.method in ['tenet']:
            self.user_adj_mat = dataset.user_adj_mat
            self.list_adj_mat = dataset.list_adj_mat
            self.item_adj_mat = dataset.item_adj_mat

        # new ================
        self.warm_start_gnn = args.warm_start_gnn ##warm-start for hgnn
        self.include_hgnn   = args.include_hgnn
        self.include_hgnn   = True if args.include_hgnn == 'True' else False

    def get_args_to_string(self,):
        args_str = str(random.randint(1,1000000))
        return args_str
