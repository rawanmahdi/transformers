import torch
import torch.nn as nn 
from torch.nn import functional as F

class Head(nn.Module):

    """ single head of self-attention """

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', tensor=torch.tril(torch.ones(block_size, block_size))) # adding as a constant parameter
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C

        # compute the affinities i.e. the attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # B,T,C multiplied by B, C, T produces B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # size is B,T,T
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x) # B,T,C
        weighted_aggregation = wei @ v # B,T,T multiplied by B,T,C produces B,T,C

        return weighted_aggregation





class MultiHeadAttention(nn.Module):

    """ multiple heads of self attention in parallel """

    def __init__(self, n_heads, block_size, head_size, dropout, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output