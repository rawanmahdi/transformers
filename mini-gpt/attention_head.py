import torch
import torch.nn as nn 
from torch.nn import functional as F

# Hyperparameters 
batch_size = 4 # number of independent sequences processed in parallel
block_size = 8 # maximum context length for prediction
max_iterations = 3000
learning_rate = 1e-2
eval_interval = 300
eval_iters = 200
n_embed = 32 # number of embedding dimensions 


class Head(nn.Module):

    """ single head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', tensor=torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C

        # compute the affinities i.e. the attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # B,T,C multiplied by B, C, T produces B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # size is B,T,T

        # perform weighted aggregation of values
        v = self.value(x) # B,T,C
        weighted_aggregation = wei @ v # B,T,T multiplied by B,T,C produces B,T,C

        return weighted_aggregation





