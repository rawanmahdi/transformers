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
        print(f"when the batched input is {context.tolist()}, the batched target: {target}")
# %%
