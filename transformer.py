import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
            Parameters
            embed_size, embedding size, e.g. 256
            heads, number of parts to split, 8 
        '''
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert(self.head_dim * heads == embed_size), 'Embedding size need to be divisible by heads'
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        values = 