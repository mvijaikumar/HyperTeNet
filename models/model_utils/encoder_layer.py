import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math

from models.model_utils.multi_head_attention import MultiHeadAttention
from models.model_utils.positionwise_feed_forward import PositionwiseFeedForward
from models.model_utils.scaled_dot_product_attention import ScaledDotProductAttention
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderLayer(nn.Module):
    '''A self-attention layer + 2 layered pff'''
    
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul,
            dropout_pff,
            diag_mask,
            bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.mul_head_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout_mul,
            diag_mask=diag_mask,
            input_dim=bottle_neck)
        self.pff_n1 = PositionwiseFeedForward(
            [d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_n2 = PositionwiseFeedForward(
            [bottle_neck, d_model, d_model], dropout=dropout_pff, residual=False, layer_norm=True)
    
    # self.dropout = nn.Dropout(0.2)
    
    def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
        dynamic, static1, attn = self.mul_head_attn(
            dynamic, dynamic, static, slf_attn_mask)
        dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
        static1 = self.pff_n2(static * non_pad_mask) * non_pad_mask

        return dynamic, static1, attn
