import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def masked_softmax(self, vector: torch.Tensor,
                       mask: torch.Tensor,
                       dim: int = -1,
                       memory_efficient: bool = False,
                       mask_fill_value: float = -1e32) -> torch.Tensor:
        
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside
                # the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).bool(), mask_fill_value) ##mine: .byte() to bool()
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result
    
    def forward(self, q, k, v, diag_mask, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask, -float('inf'))
        
        attn = self.masked_softmax(
            attn, diag_mask, dim=-1, memory_efficient=True)
        
        
        output = torch.bmm(attn, v)
        
        return output, attn

