import torch.nn as nn
from attention_head import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):

    """ transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

