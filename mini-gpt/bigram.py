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
torch.manual_seed(1337)


with open('input_data', 'r', encoding='utf-8') as f:
    text = f.read()


# get all unique characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# create tokenizer fpr the input text by characters
str2int = { ch:i for i,ch in enumerate(vocab)}
int2str = { i:ch for i,ch in enumerate(vocab)}
encoder = lambda s: [str2int[c] for c in s] # given string, return list of ints 
decorder = lambda l: ''.join([int2str[i] for i in l]) # given list, return string

# encode input dataset and place it in tensor 
data = torch.tensor(encoder(text), dtype=torch.long)

# train validation split 
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # generate $batch_size number of random indexes
    x = torch.stack([data[i:i+block_size] for i in ix]) # batch chunks for each random index 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 for next char prediction
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 
xb, yb = get_batch('train')

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
    
model = BigramLanguageModel(vocab_size=vocab_size)

# get pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iterations):

    # evaluate loss on training and validation data every few iterations 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Training loss at step {iter}: {losses['train']:.4f} \nValidation loss at step {iter}: {losses['val']:.4f}")

    # sample a batch
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate a sample output
context = torch.zeros((1,1), dtype=torch.long)
print(decorder(model.generate(context, max_new_tokens=500)[0].tolist()))
