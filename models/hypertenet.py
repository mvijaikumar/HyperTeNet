import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from models.hypersagnn import HyperSAGNN
from models.my_gcn_conv import GCNConv
from models.transformer_model import TransformerModel
from utils.gnn_utils import normalize_adj


class HyperTeNet(torch.nn.Module):
    def __init__(self, params, device='cuda:0'):
        super(HyperTeNet, self).__init__()
        self.params                      = params

        # embedding matrices
        self.user_list_item_embeddings   = torch.nn.Embedding(params.num_user + params.num_list + params.num_item, params.num_factors)
        self.fc1                         = torch.nn.Linear(params.num_factors, 1)
        self.fc2                         = torch.nn.Linear(params.num_factors, 1)
        self.fc3                         = torch.nn.Linear(params.hid_units[-1], 1)
        self.fc4                         = torch.nn.Linear(params.hid_units[-1], 1)

        self.user_item_list_dropout      = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout1                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout2                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.sigmoid                     = torch.nn.Sigmoid()

        # weight initialization
        ##torch.nn.init.xavier_uniform_(self.user_list_item_embeddings.weight)
        torch.nn.init.xavier_normal_(self.user_list_item_embeddings.weight)
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[0]) ## check in the successive iterations that this is kept zero
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user])
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user + self.params.num_list])

        # gnn ==========================
        self.user_indices           = torch.tensor(np.array(range(params.num_user))).to(device).long()
        self.list_indices           = torch.tensor(np.array(range(params.num_list))).to(device).long()
        self.item_indices           = torch.tensor(np.array(range(params.num_item))).to(device).long()

        self.user_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True) ##normalize=True
        self.user_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)

        self.list_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True)
        self.list_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)

        self.item_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True)
        self.item_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)


        if params.args.knn_graph == 'True':
            self.user_param_indices     = params.dataset_obj.user_edge_index
            self.list_param_indices     = params.dataset_obj.list_edge_index
            self.item_param_indices     = params.dataset_obj.item_edge_index
            self.user_param_weights, self.list_param_weights, self.item_param_weights = None, None, None ##crucial to note
        else:
            self.user_adj_mat           = params.user_adj_mat.tocoo()
            self.user_adj_mat.setdiag(0); self.user_adj_mat.eliminate_zeros()
            #pdb.set_trace()
            self.user_param_indices     = torch.LongTensor(self.user_adj_mat.nonzero()).to(device)
            self.user_param_weights     = torch.FloatTensor(self.user_adj_mat.data).to(device) ##weight check

            self.list_adj_mat           = params.list_adj_mat.tocoo()
            self.list_adj_mat.setdiag(0); self.list_adj_mat.eliminate_zeros()
            self.list_param_indices     = torch.LongTensor(self.list_adj_mat.nonzero()).to(device)
            self.list_param_weights     = torch.FloatTensor(self.list_adj_mat.data).to(device) ##weight check

            self.item_adj_mat           = params.item_adj_mat.tocoo()
            self.item_adj_mat.setdiag(0); self.item_adj_mat.eliminate_zeros()
            self.item_param_indices     = torch.LongTensor(self.item_adj_mat.nonzero()).to(device)
            self.item_param_weights     = torch.FloatTensor(self.item_adj_mat.data).to(device) ##weight check
            if params.args.user_adj_weights == 'False':
                self.user_param_weights, self.list_param_weights, self.item_param_weights = None, None, None ##crucial to note

        # dropouts gnn part
        self.user_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob
        self.list_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob
        self.item_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob

        # seq part ============================================================================
        self.pos_embeddings                  = torch.nn.Embedding(params.max_item_seq_length, params.hid_units[-1])

        self.user_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.list_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout1                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout2                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout3                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout4                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout5                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout6                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout7                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.pos_dropout                     = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.sigmoid_seq                     = torch.nn.Sigmoid()

        # transormer model ======================
        self.trans_model                     = TransformerModel(ntoken=params.num_item, ninp=params.hid_units[-1], nhead=params.n_heads[-1],
                                                                    nhid=params.hid_units[-1], nlayers=params.num_layers, dropout=0.3)
        self.layer_norm                      = nn.LayerNorm(params.hid_units[-1])

        # hgnn ==================================
        self.hypersagnn_model                = HyperSAGNN(n_head=params.n_heads[0], d_model=params.hid_units[-1], d_k=params.hid_units[-1], d_v=params.hid_units[-1],
                                                 node_embedding=self.user_list_item_embeddings,
                                                 diag_mask=True, bottle_neck=params.hid_units[-1],
                                                 dropout=1.0-params.net_keep_prob).to(device)



    def get_emb_user(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        #emb    = self.user_item_list_dropout(emb)
        output = emb[:,0] * emb[:,2] #user-item
        #output = emb[:,1] * emb[:,2] #list-item
        #output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.user_item_list_dropout(output)
        #output = self.sigmoid(torch.sum(output,axis=1)) #self.user_item_list_dropout(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_list(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        #emb    = self.user_item_list_dropout(emb)
        output = emb[:,1] * emb[:,2] #user-item
        #output = emb[:,1] * emb[:,2] #list-item
        #output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.user_item_list_dropout(output)
        #output = self.sigmoid(torch.sum(output,axis=1)) #self.user_item_list_dropout(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_all_mult(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.dropout1(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_all_mult2(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.dropout1(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list2(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = user_list_item_embeddings[x]
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        #output = self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list3(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = user_list_item_embeddings[x]
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(self.fc3(output_user).reshape(-1) + self.fc4(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def forward(self, user_indices, list_indices, item_indices=None, item_seq=None, item_seq_pos=None, item_seq_neg=None, test_item_indices=None, param5=None, train=True, network='gnn', include_hgnn=False):
    #def forward(self, user_indices, list_indices, item_seq, item_seq_pos=None, item_seq_neg=None, test_item_indices=None, param5=None, train=True,network='seq'):

        # gnn_user ==============================
        user_x = self.user_list_item_embeddings(self.user_indices.long())
        user_x = F.relu(self.user_conv1(user_x, self.user_param_indices, self.user_param_weights))
        user_x = self.user_gnn_dropout(user_x)
        user_x = self.user_conv2(user_x, self.user_param_indices, self.user_param_weights)

        # gnn_list ==============================
        list_x = self.user_list_item_embeddings(self.params.num_user+self.list_indices.long())
        list_x = F.relu(self.list_conv1(list_x, self.list_param_indices, self.list_param_weights))
        list_x = self.list_gnn_dropout(list_x)
        list_x = self.list_conv2(list_x, self.list_param_indices, self.list_param_weights)

        # gnn_item ==============================
        item_x = self.user_list_item_embeddings(self.params.num_user+self.params.num_list+self.item_indices.long())
        item_x = F.relu(self.item_conv1(item_x, self.item_param_indices, self.item_param_weights))
        item_x = self.item_gnn_dropout(item_x)
        item_x = self.item_conv2(item_x, self.item_param_indices, self.item_param_weights)

        user_list_item_gnn_emb = torch.cat([user_x, list_x, item_x],dim=0)
        # common part ending ========================================================================

        # gnn ===================================
        if network == 'gnn':
            x = torch.cat([user_indices.reshape(-1,1),
                       list_indices.reshape(-1,1) + self.params.num_user,
                       item_indices.reshape(-1,1) + self.params.num_user + self.params.num_list],
                       dim=1)


            self.edge_probs_gnn        = self.get_emb_user_list3(x,user_list_item_gnn_emb)

            # hgnn =======================
            if include_hgnn == True:
                self.edge_probs_hgnn   = self.hypersagnn_model(x, user_list_item_gnn_emb).reshape(-1)
                self.edge_probs        = (self.edge_probs_hgnn + self.edge_probs_gnn)/2
            else:
                self.edge_probs        = self.edge_probs_gnn
            ##self.edge_probs          = self.edge_probs_hgnn
            ##self.edge_probs          = self.edge_probs_gnn
            return self.edge_probs

        # seq ===================================
        elif network == 'seq':
            flag_tran = True
            if train == False:
                user_indices                     = user_indices.reshape(-1,101)[:,0]
                list_indices                     = list_indices.reshape(-1,101)[:,0]
                item_seq                         = item_seq.reshape(-1,101,self.params.max_item_seq_length)[:,0,:] ##101

            self.user_embeds                     = user_x[user_indices]
            self.list_embeds                     = list_x[list_indices]
            self.user_embeds                     = self.user_dropout(self.user_embeds)
            self.list_embeds                     = self.list_dropout(self.list_embeds)
            self.mask                            = (item_seq != 0).float()

            self.item_seq_embeds                 = item_x[item_seq]
            self.item_seq_embeds                += (self.user_embeds.reshape(-1,1,self.params.hid_units[-1]) + self.list_embeds.reshape(-1,1,self.params.hid_units[-1]))
            self.item_seq_embeds                += self.pos_embeddings.weight ##check this carefullly
            self.item_seq_embeds                *= self.mask.reshape(item_seq.shape[0], item_seq.shape[1], 1)

            if flag_tran == True:
                self.out_trans                   = self.trans_model(self.item_seq_embeds.transpose(1,0)).transpose(1,0) ##posemb
                self.item_seq_embeds             = self.out_trans

            if train == True:
                self.item_seq_pos_embeds         = item_x[item_seq_pos]
                self.item_seq_neg_embeds         = item_x[item_seq_neg]
                self.item_seq_pos_embeds         = self.item_dropout3(self.item_seq_pos_embeds)
                self.item_seq_neg_embeds         = self.item_dropout4(self.item_seq_neg_embeds)
                self.is_target                   = (item_seq_pos != 0).float()

                self.user_item_seq_pos_embeds    = self.user_embeds.reshape(-1,1,self.params.hid_units[-1]) * self.item_seq_pos_embeds
                self.list_item_seq_pos_embeds    = self.list_embeds.reshape(-1,1,self.params.hid_units[-1]) * self.item_seq_pos_embeds
                self.item_seq_and_seq_pos_embeds = self.item_seq_embeds * self.item_seq_pos_embeds

                self.user_item_seq_neg_embeds    = self.user_embeds.reshape(-1,1,self.params.hid_units[-1]) * self.item_seq_neg_embeds
                self.list_item_seq_neg_embeds    = self.list_embeds.reshape(-1,1,self.params.hid_units[-1]) * self.item_seq_neg_embeds
                self.item_seq_and_seq_neg_embeds = self.item_seq_embeds * self.item_seq_neg_embeds

                self.pos_logits              = self.sigmoid(torch.sum(self.user_item_seq_pos_embeds + self.list_item_seq_pos_embeds + self.item_seq_and_seq_pos_embeds, axis=-1))
                self.neg_logits              = self.sigmoid(torch.sum(self.user_item_seq_neg_embeds + self.list_item_seq_neg_embeds + self.item_seq_and_seq_neg_embeds, axis=-1))

                return self.pos_logits, self.neg_logits, self.is_target

            elif train == False:
                self.test_item_embeds            = item_x[test_item_indices]
                self.item_seq_embeds             = self.item_seq_embeds.view(-1,1,self.params.max_item_seq_length,self.params.hid_units[-1]).repeat(1,101,1,1).view(-1,self.params.max_item_seq_length,self.params.hid_units[-1])
                self.list_embeds                 = self.list_embeds.view(-1,1,self.params.hid_units[-1]).repeat(1,101,1).view(-1,self.params.hid_units[-1])
                self.user_embeds                 = self.user_embeds.view(-1,1,self.params.hid_units[-1]).repeat(1,101,1).view(-1,self.params.hid_units[-1])

                self.pos_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds[:,-1,:] + self.list_embeds + self.user_embeds) * self.test_item_embeds,axis=-1))
                return self.pos_logits

