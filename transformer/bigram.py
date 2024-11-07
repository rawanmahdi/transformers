import torch
import torch.nn as nn
from torch.nn import functional as F
from attention_head import Head, MultiHeadAttention
from feed_forward import FeedForward
from transformer_block import TransformerBlock
# Hyperparameters 
batch_size = 4 # number of independent sequences processed in parallel
block_size = 128 # maximum context length for prediction
n_embed = 142 # number of embedding dimensions 
n_heads = 4
n_layers = 6
dropout = 0.2
vocab_size = 65

# for reproducible results 
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # this will allow us to encode the meaning of the tokens 
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # this will allow us to encode the position of the tokens
        # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embed//4)
        # self.ffwd = FeedForward(n_embed=n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # finaly layer normalization 
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        position_embedding = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_embeddings + position_embedding # (B,T,C)
        # x = self.sa_heads(x) # apply one head of self attention
        # x = self.ffwd(x) # (B,T,C) - this layer allows the model to 'think' on the context before producing the logits
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None: # in the case where we are running inference
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # convert array to 2Dtr
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices for the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens 
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # get last time step, (B,C)
            probs = F.softmax(logits, dim=1) # get probability 
            next_idx = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx
    
