import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

class Wrap_Embedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *input):
        return super().forward(*input), torch.Tensor([0]).to(device)

# Used only for really big adjacency matrix


class SparseEmbedding(nn.Module):
    def __init__(self, embedding_weight, sparse=True):
        super().__init__()
        print(embedding_weight.shape)
        self.sparse = sparse
        if self.sparse:
            self.embedding = embedding_weight
        else:
            try:
                try:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight.todense())).to(device)
                except BaseException:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight)).to(device)
            except Exception as e:
                print("Sparse Embedding Error",e)
                self.sparse = True
                self.embedding = embedding_weight
    
    def forward(self, x):
        
        if self.sparse:
            x = x.cpu().numpy()
            x = x.reshape((-1))
            temp = np.asarray((self.embedding[x, :]).todense())
            
            return torch.from_numpy(temp).to(device)
        else:
            return self.embedding[x, :]

class TiedAutoEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(out, inp))
        self.bias1 = nn.parameter.Parameter(torch.Tensor(out))
        self.bias2 = nn.parameter.Parameter(torch.Tensor(inp))
        
        self.register_parameter('tied weight',self.weight)
        self.register_parameter('tied bias1', self.bias1)
        self.register_parameter('tied bias2', self.bias2)
        
        self.reset_parameters()
        
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias1, -bound, bound)
        
        if self.bias2 is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias2, -bound, bound)

    def forward(self, input):
        encoded_feats = F.linear(input, self.weight, self.bias1)
        encoded_feats = F.tanh(encoded_feats)
        reconstructed_output = F.linear(encoded_feats, self.weight.t(), self.bias2)
        return encoded_feats, reconstructed_output


class Word2vec_Skipgram(nn.Module):
    def __init__(
            self,
            dict_size,
            embedding_dim,
            window_size,
            u_embedding=None,
            sparse=False):
        super(Word2vec_Skipgram, self).__init__()
        '''
        use context (u) to predict center (v)
        '''
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        self.u_embedding = u_embedding
        self.sm_w_t = nn.Embedding(
            dict_size,
            embedding_dim,
            sparse=sparse,
            padding_idx=0,
        )
        self.sm_b = nn.Embedding(dict_size, 1, sparse=sparse, padding_idx=0, )
    
    def forward_u(self, u):
        return self.u_embedding(u)
    
    def forward_w_b(self, id):
        return self.sm_w_t(id), self.sm_b(id)

# utility functions

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk
    
    return padding_mask



