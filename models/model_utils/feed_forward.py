# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb


class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    
    def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
        super(FeedForward, self).__init__()
        self.w_stack = []
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
            self.add_module("FF_Linear%d" % (i), self.w_stack[-1])
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.reshape = reshape
    
    def forward(self, x):
        output = x
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        
        return output

