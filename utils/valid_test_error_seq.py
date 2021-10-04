import numpy as np
import torch
import math
from utils.evaluation import evaluate_model
from time import time
from utils.batch import Batch


class ValidTestErrorSEQ(object):
    def __init__(self, params):
        self.params               = params
        self.validNegativesDict   = params.validNegativesDict
        self.testNegativesDict    = params.testNegativesDict

        self.num_valid_instances  = params.num_valid_instances
        self.num_test_instances   = params.num_test_instances
        self.num_thread           = params.num_thread
        self.num_valid_negatives  = self.get_num_valid_negative_samples(self.validNegativesDict)
        self.valid_dim            = self.num_valid_negatives + 1

        self.epoch_mod            = params.epoch_mod
        self.valid_batch_siz      = params.valid_batch_siz
        self.at_k                 = params.at_k

        self.validArrDubles,self.valid_pos_items = self.get_dict_to_dubles(self.validNegativesDict)
        self.testArrDubles,self.test_pos_items   = self.get_dict_to_dubles(self.testNegativesDict)
        self.list_user_vec                       = params.list_user_vec

    def get_num_valid_negative_samples(self,validDict): ## some strange things are happening.
        for key in validDict:
            return len(self.validNegativesDict[key])
        return None

    def get_dict_to_dubles(self,dct):
        list_lst, item_lst = [],[]
        pos_item_lst = []
        for key,value in dct.items():
            lst_id, itm_id = key
            lists  = list(np.full(self.valid_dim,lst_id,dtype = 'int32'))#+1 to add pos item
            items  = [itm_id]
            pos_item_lst.append(itm_id)
            items += list(value) # first is positive item

            list_lst   += lists
            item_lst   += items

        return (np.array(list_lst),np.array(item_lst)),np.array(pos_item_lst)

    def get_update(self,model,epoch_num,device,valid_flag=True):
        model.eval()
        if valid_flag == True:
            (list_input,item_input) = self.validArrDubles
            num_inst   = self.num_valid_instances * self.valid_dim
            posItemlst = self.valid_pos_items # parameter for evaluate_model
            matShape   = (self.num_valid_instances, self.valid_dim)
        else:
            (list_input,item_input) = self.testArrDubles
            num_inst   = self.num_test_instances * self.valid_dim
            posItemlst = self.test_pos_items # parameter for evaluate_model
            matShape   = (self.num_test_instances, self.valid_dim)

        batch_siz      = self.valid_batch_siz * self.valid_dim

        full_pred_torch_lst  = []
        list_input_ten       = torch.from_numpy(list_input.astype(np.long)).to(device) ## could be moved to gpu before-hand
        item_input_ten       = torch.from_numpy(item_input.astype(np.long)).to(device)
        user_input           = self.list_user_vec[list_input]
        user_input_ten       = torch.from_numpy(user_input.astype(np.long)).to(device)
        batch                = Batch(num_inst,batch_siz,shuffle=False)
        while batch.has_next_batch():
            batch_indices    = batch.get_next_batch_indices()

            if valid_flag == True:
                item_seq         = torch.from_numpy(self.params.train_matrix_item_seq[list_input[batch_indices]].astype(np.long)).to(device) ## ##for_test
            else:
                item_seq         = torch.from_numpy(self.params.train_matrix_item_seq_for_test[list_input[batch_indices]].astype(np.long)).to(device) ## ##for_test
            y_pred           = model(user_indices=user_input_ten[batch_indices],list_indices=list_input_ten[batch_indices],item_seq=item_seq,
                                     test_item_indices=item_input_ten[batch_indices],train=False,network='seq') # ##
            full_pred_torch_lst.append(y_pred.detach().cpu().numpy())

        full_pred_np         = np.concatenate(full_pred_torch_lst) #.data.cpu().numpy()
        # ==============================

        predMatrix           = np.array(full_pred_np).reshape(matShape)
        itemMatrix           = np.array(item_input).reshape(matShape)

        (hits, ndcgs, maps)  = evaluate_model(posItemlst=posItemlst,itemMatrix=itemMatrix,predMatrix=predMatrix,k=self.at_k,num_thread=self.num_thread)
        return (hits, ndcgs, maps)
