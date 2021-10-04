import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

# A custom position-wise MLP.
# dims is a list, it would create multiple layer with tanh between them
# If dropout, it would add the dropout at the end. Before residual and
# layer-norm


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            dims,
            dropout=None,
            reshape=False,
            use_bias=True,
            residual=False,
            layer_norm=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, use_bias))
            self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[-1])
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.residual = residual
        self.layer_norm_flag = layer_norm
    
    def forward(self, x):
        output = x.transpose(1, 2)
        
        
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        
        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)
        
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        
        if self.dims[0] == self.dims[-1]:
            # residual
            if self.residual:
                output += x

            if self.layer_norm_flag:
                output = self.layer_norm(output)
        
        return output
