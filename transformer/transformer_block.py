import torch.nn as nn
from attention_head import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):

    """ transformer block: communication followed by computation """

    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, block_size, head_size, dropout, n_embed)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

