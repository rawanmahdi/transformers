import torch
from bigram import BigramLanguageModel

batch_size = 4 # number of independent sequences processed in parallel
block_size = 128 # maximum context length for prediction
max_iterations = 5000
learning_rate = 3e-4
eval_interval = 1000
eval_iters = 200

with open('../input_data', 'r', encoding='utf-8') as f:
    text = f.read()


# get all unique characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(vocab_size)

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

@torch.no_grad() #no back propagation! more efficient memory mode
def estimate_loss():
    out = {}
    model.eval # switch to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # switch back to train mode
    return out 
xb, yb = get_batch('train')


model = BigramLanguageModel()

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

