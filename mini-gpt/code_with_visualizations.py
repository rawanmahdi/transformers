#%%
!curl https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt -o input_data
# %%
with open('input_data', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in chars:", len(text))
print("first thousand chars: \n", text[:1000])
# %%

# get all unique characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print('vocab size: ',vocab_size)
print('vocab: ',''.join(vocab))
# %%

# tokenize the input text by characters

str2int = { ch:i for i,ch in enumerate(vocab)}
int2str = { i:ch for i,ch in enumerate(vocab)}

encoder = lambda s: [str2int[c] for c in s] # given string, return list of ints 
decorder = lambda l: ''.join([int2str[i] for i in l]) # given list, return string

print(encoder("my name is rawamily"))
print(decorder(encoder("my name is rawamily")))
# %%
import torch


# encode input dataset and place it in tensor 
data = torch.tensor(encoder(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
# %%
# train validation split 
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

#%%
block_size = 8
train_data[:block_size+1]

# illustration of prediction based on full context of the block:
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target is {target}")
# %%
# batching for parallel processing of blocks 

torch.manual_seed(1337)
batch_size = 4 # number of independent sequences processed in parallel
block_size = 8 # maximum context length for prediction

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # generate $batch_size number of random indexes
    x = torch.stack([data[i:i+block_size] for i in ix]) # batch chunks for each random index 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 for next char prediction
    return x, y

xb, yb = get_batch('train')
print('inputs: \n', xb.shape, "\n", xb)
print('targets: \n', yb.shape, "\n", yb)

#%%
# inputs vs targets illustration
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        print(context)
        target = yb[b, t]
        print(f"when the input is {context.tolist()}, the target: {target}")
# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None: # in the case where we are running inference
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # convert array to 2D
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target=targets)

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices for the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :] # get last time step, (B,C)
            probs = F.softmax(logits, dim=1) # get probability 
            next_idx = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decorder(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# %%
# get pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
#%%
# training loop
batch_size = 32
for steps in range(100000):
    # sample a batch
    xb, yb = get_batch('train')
    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())
# %%
print(decorder(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# %%
