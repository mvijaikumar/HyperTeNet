import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math
import pdb
import sys

from models.model_utils.encoder_layer import EncoderLayer
from models.model_utils.scaled_dot_product_attention import ScaledDotProductAttention
from models.model_utils.feed_forward import FeedForward
from models.model_utils.multi_head_attention import MultiHeadAttention
from models.model_utils.positionwise_feed_forward import PositionwiseFeedForward
from models.model_utils.hypersagnn_utils import get_attn_key_pad_mask, get_non_pad_mask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HyperSAGNN(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            node_embedding,
            diag_mask,
            bottle_neck,
            dropout
    ):
        super().__init__()
        
        self.pff_classifier = PositionwiseFeedForward(
            [d_model, 1], reshape=True, use_bias=True)
        
        self.node_embedding = node_embedding
        #self.encode1 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.3, dropout_pff=0.4, diag_mask=diag_mask, bottle_neck=bottle_neck)
        ## mine 
        ##self.encode1 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.6, dropout_pff=0.6, diag_mask=diag_mask, bottle_neck=bottle_neck)
        #self.encode1 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.0, dropout_pff=0.0, diag_mask=diag_mask, bottle_neck=bottle_neck)
        self.encode1 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.4, dropout_pff=0.4, diag_mask=diag_mask, bottle_neck=bottle_neck)
        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        #pdb.set_trace()
        self.dropout_m1 = nn.Dropout(dropout)
    
    def get_node_embeddings(self, x, user_list_item_embeds, return_recon = False): ##treat embeds as embeddings table
        ## Mine: This is the place I have to modify the code to get the gmf embeddings 1. concatenate, 2. individual for user, item and list
        
        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        
        x = user_list_item_embeds[x.view(-1)]
        ##x = self.node_embedding[x.view(-1)]
        ##x, recon_loss = self.node_embedding(x.view(-1))
        ##x = self.node_embedding(x.view(-1))
        if return_recon:
            #return x.view(sz_b, len_seq, -1), recon_loss
            #return user_list_item_embeds, recon_loss
            return x.view(sz_b, len_seq, -1), recon_loss
        else:
            #return user_list_item_embeds
            return x.view(sz_b, len_seq, -1)
    
    def get_embedding(self, x, user_list_item_embeds,slf_attn_mask, non_pad_mask,return_recon = False):
        if return_recon:
            x, recon_loss = self.get_node_embeddings(x,user_list_item_embeds,return_recon)
        else:
            x = self.get_node_embeddings(x, user_list_item_embeds,return_recon)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static1, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if return_recon:
            return dynamic, static, attn, recon_loss
        else:
            return dynamic, static, attn
    
    def get_embedding_static(self, x):
        if len(x.shape) == 1:
            x = x.view(-1, 1)
            flag = True
        else:
            flag = False
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        x = self.get_node_embeddings(x)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if flag:
            return static[:, 0, :]
        return static
    
    def forward(self, x, user_list_item_embeds, mask=None, get_outlier=None, return_recon = False):
        x = x.long()
            
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        
        if return_recon:
            dynamic, static, attn, recon_loss = self.get_embedding(x,user_list_item_embeds, slf_attn_mask, non_pad_mask,return_recon)
        else:
            dynamic, static, attn = self.get_embedding(x, user_list_item_embeds,slf_attn_mask, non_pad_mask, return_recon)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)
        sz_b, len_seq, dim = dynamic.shape
        
        if self.diag_mask_flag == 'True':
            output = (dynamic - static) ** 2
        else:
            output = dynamic

        ## change this back to original
        #output = static
        #output = dynamic
        
        ## mine
        #output = self.dropout_m1(output)

        output = self.pff_classifier(output)
        output = torch.sigmoid(output) ##sigmoid should be there if BECwithLogits not used
        ##output = torch.relu(output)
        
        
        if get_outlier is not None:
            k = get_outlier
            outlier = (
                    (1 -
                     output) *
                    non_pad_mask).topk(
                k,
                dim=1,
                largest=True,
                sorted=True)[1]
            return outlier.view(-1, k)
        
        mode = 'sum'
        #mode = 'first'
        
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output
        
        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum
        elif mode == 'first':
            output = output[:, 0, :]
            
        if return_recon:
            return output, recon_loss
        else:
            return output
