import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

class MultipleEmbedding(nn.Module):
    def __init__(
            self,
            embedding_weights,
            dim,
            sparse=True,
            num_list=None,
            node_type_mapping=None):
        super().__init__()
        print(dim)
        self.num_list = torch.tensor([0] + list(num_list)).to(device)
        print(self.num_list)
        self.node_type_mapping = node_type_mapping
        self.dim = dim
        
        self.embeddings = []
        for i, w in enumerate(embedding_weights):
            try:
                self.embeddings.append(SparseEmbedding(w, sparse))
            except BaseException as e:
                print ("Conv Embedding Mode")
                self.add_module("ConvEmbedding1", w)
                self.embeddings.append(w)
        
        test = torch.zeros(1, device=device).long()
        self.input_size = []
        for w in self.embeddings:
            self.input_size.append(w(test).shape[-1])
        
        self.wstack = [TiedAutoEncoder(self.input_size[i],self.dim).to(device) for i,w in enumerate(self.embeddings)]
        self.norm_stack =[nn.LayerNorm(self.dim).to(device) for w in self.embeddings]
        for i, w in enumerate(self.wstack):
            self.add_module("Embedding_Linear%d" % (i), w)
            self.add_module("Embedding_norm%d" % (i), self.norm_stack[i])
            
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        
        final = torch.zeros((len(x), self.dim)).to(device)
        recon_loss = torch.Tensor([0.0]).to(device)
        for i in range(len(self.num_list) - 1):
            select = (x >= (self.num_list[i] + 1)) & (x < (self.num_list[i + 1] + 1))
            if torch.sum(select) == 0:
                continue
            adj = self.embeddings[i](x[select] - self.num_list[i] - 1)
            output = self.dropout(adj)
            output, recon = self.wstack[i](output)
            output = self.norm_stack[i](output)
            final[select] = output
            recon_loss += sparse_autoencoder_error(recon, adj)
            
        return final, recon_loss

def sparse_autoencoder_error(y_pred, y_true):
    return torch.mean(torch.sum((y_true.ne(0).type(torch.float) * (y_true - y_pred)) ** 2, dim = -1) / torch.sum(y_true.ne(0).type(torch.float), dim = -1))



