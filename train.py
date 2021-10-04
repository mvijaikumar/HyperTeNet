import sys
import math
import argparse
import os
import pdb
import random
import datetime
import warnings
from time import time
from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data.dataset import Dataset
from data.tenet_dataset import TenetDataset
from data.embed_dataset import EmbedDataset
from models.models_base import Models
from utils.arguments import parse_args
from utils.parameters import Parameters
from utils.tenet_negative_samples import NegativeSamples
from utils.list_negative_samples import ListNegativeSamples
from utils.batch import Batch
from utils.valid_test_error import ValidTestError
from utils.valid_test_error_seq import ValidTestErrorSEQ
from utils.evaluation import evaluate_model
from utils.sampler import WarpSampler
from utils.checkpoint import save_model
from utils import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parse_args()
    print('Loading dataset...', end="")
    args.device = device
    args.date_time = datetime.datetime.now()

    dataset = EmbedDataset(args)
    params = Parameters(args, dataset)
    print("\rDataset Statistics:")
    print(f"    Users: {params.num_user} | Lists: {params.num_list} | Items:{params.num_item}")
    print(f"    Train: {params.num_train_instances} | Valid: {params.num_valid_instances} | Test: {params.num_test_instances}")
    print(f"    Density: {100 * params.num_train_instances / (params.num_list * params.num_item):.4f} %")
    print("="*60)
    args.args_str = params.get_args_to_string()
    t1 = time()

    models = Models(params, device=device)
    model = models.get_model()
    model.to(device)
    save_model_path = os.path.join("./saved_models", params.dataset + ".pth.tar")

    criterion_li = torch.nn.BCELoss()
    optimizer_gnn = torch.optim.Adam(model.parameters(), lr=params.lr)
    optimizer_seq = torch.optim.Adam(model.parameters(), lr=params.lr)

    vt_err_gnn = ValidTestError(params)
    vt_err_seq = ValidTestErrorSEQ(params)

    include_networks = eval(args.include_networks)
    valid_type = 'seq'
    vt_err = vt_err_gnn if valid_type == 'gnn' else vt_err_seq
    ns_gnn = NegativeSamples(params.train_matrix, params.num_negatives, params)
    ns_seq = ListNegativeSamples(params.train_matrix_item_seq, params.num_negatives, params)

    best_ndcg = -1
    include_hgnn_flag = True
    for epoch_num in range(params.num_epochs + 1):
        tt = time()
        model.train()
        for network in include_networks:
            t2 = time()
            ce_or_pairwise_loss, reg_loss, recon_loss = 0.0, 0.0, 0.0
            if network == 'gnn':
                if epoch_num > params.warm_start_gnn:
                    params.epoch_mod = 1
                user_input, list_input, item_input, train_rating = ns_gnn.generate_instances()
                user_input, list_input, item_input, train_rating = (
                    torch.from_numpy(user_input.astype(np.long)).to(device),
                    torch.from_numpy(list_input.astype(np.long)).to(device),
                    torch.from_numpy(item_input.astype(np.long)).to(device),
                    torch.from_numpy(train_rating.astype(np.float32)).to(device)
                )
                num_inst = len(user_input)
            elif network == 'seq':
                list_input, item_seq, item_seq_pos, item_seq_neg = ns_seq.generate_instances()
                user_input = params.list_user_vec[list_input]
                user_input, list_input, item_seq, item_seq_pos, item_seq_neg = (
                    torch.from_numpy(user_input.astype(np.long)).to(device),
                    torch.from_numpy(list_input.astype(np.long)).to(device),
                    torch.from_numpy(item_seq.astype(np.long)).to(device),
                    torch.from_numpy(item_seq_pos.astype(np.long)).to(device),
                    torch.from_numpy(item_seq_neg.astype(np.long)).to(device)
                )
                num_inst = len(list_input)

            if network == 'gnn' and params.loss not in ['bpr']:
                batch = Batch(num_inst, params.batch_size, shuffle=True)
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_gnn.zero_grad()

                    y_pred = model(user_indices=user_input[batch_indices], list_indices=list_input[batch_indices],
                                   item_indices=item_input[batch_indices], network=network,
                                   include_hgnn=include_hgnn_flag)
                    y_orig = train_rating[batch_indices]
                    loss = criterion_li(y_pred, y_orig)
                    loss.backward()
                    optimizer_gnn.step()
                    ce_or_pairwise_loss += loss * len(batch_indices)
            elif network == 'seq':
                batch = Batch(num_inst, params.batch_size_seq, shuffle=True)
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_seq.zero_grad()
                    y_pred_seq_pos, y_pred_seq_neg, is_target = model(user_indices=user_input[batch_indices].long(),
                                                                      list_indices=list_input[batch_indices].long(),
                                                                      item_seq=item_seq[batch_indices].long(),
                                                                      item_seq_pos=item_seq_pos[batch_indices].long(),
                                                                      item_seq_neg=item_seq_neg[batch_indices].long(),
                                                                      train=True, network=network)
                    first_flag = True
                    for ind_neg in range(params.num_negatives_seq - 1):
                        neg_indices = np.arange(0, num_inst)
                        np.random.shuffle(neg_indices)
                        neg_batch_indices = neg_indices[0:len(batch_indices)]

                        _, y_pred_seq_neg_arr_local, _ = model(user_indices=user_input[batch_indices].long(),
                                                               list_indices=list_input[batch_indices].long(),
                                                               item_seq=item_seq[batch_indices].long(),
                                                               item_seq_pos=item_seq_pos[batch_indices].long(),
                                                               item_seq_neg=item_seq_neg[neg_batch_indices].long(),
                                                               train=True, network=network)
                        if first_flag:
                            first_flag = False
                            y_pred_seq_neg_sum = (1 - y_pred_seq_neg_arr_local + 1e-24).log() * is_target
                        else:
                            y_pred_seq_neg_sum += (1 - y_pred_seq_neg_arr_local + 1e-24).log() * is_target
                    if params.num_negatives_seq <= 1:
                        loss = (-(y_pred_seq_pos + 1e-24).log() * is_target - (
                                    1 - y_pred_seq_neg + 1e-24).log() * is_target).sum() / is_target.sum()
                    else:
                        loss = (-(y_pred_seq_pos + 1e-24).log() * is_target - (
                                    1 - y_pred_seq_neg + 1e-24).log() * is_target -
                                y_pred_seq_neg_sum).sum() / is_target.sum()

                    loss.backward()
                    optimizer_seq.step()
                    ce_or_pairwise_loss += loss
            # training ends =======================================================================
            total_loss = ce_or_pairwise_loss + reg_loss + recon_loss
            print("[%.2f s] %15s iter:%3i ==> Loss:%.4f " % (time() - t2, network, epoch_num, total_loss))

        # validation and test =======================================================================
        if epoch_num % params.epoch_mod == 0:
            t3 = time()
            (valid_hits_lst, valid_ndcg_lst, valid_map_lst) = vt_err.get_update(model, epoch_num, device,
                                                                                valid_flag=True)
            (test_hits_lst, test_ndcg_lst, test_map_lst) = vt_err.get_update(model, epoch_num, device, valid_flag=False)
            (valid_hr, valid_ndcg, valid_map) = (
                np.mean(valid_hits_lst), np.mean(valid_ndcg_lst), np.mean(valid_map_lst))
            (test_hr, test_ndcg, test_map) = (np.mean(test_hits_lst), np.mean(test_ndcg_lst), np.mean(test_map_lst))
            print(
                "[%.2f s] %15s Errors train %.4f valid hr: %.4f test hr: %.4f valid ndcg: %.4f test ndcg: %.4f valid map: %.4f test map: %.4f" % (
                    time() - t3, '', ce_or_pairwise_loss / num_inst, valid_hr, test_hr, valid_ndcg, test_ndcg, valid_map,
                    test_map)
            )

            print('Time taken for this epoch: {:.2f} m'.format((time() - tt) / 60))

            if test_ndcg > best_ndcg:
                best_ndcg = test_ndcg
                save_model(model.state_dict(), save_model_path)

    print("Trained!")
    print("="*60)
