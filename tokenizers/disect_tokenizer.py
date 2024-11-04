#%%
from transformers import BertModel, AutoTokenizer
import pandas as pd

#%%
model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

sentence = "When life gives you lemons, don't make lemonade"
tokenizer.tokenize(sentence)

#%%
vocab_df = pd.DataFrame({"tokens":tokenizer.vocab.keys(), "token_ID":tokenizer.vocab.values()}) 
vocab_df
# %%
model
# %%

tokernizer_output = tokenizer(sentence)
tokernizer_output

"""
{'input_ids': [101, 1332, 1297, 3114, 1128, 22782, 1116, 117, 1274, 112, 189, 1294, 22782, 6397, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
# %%
sentence2 = sentence.replace("don't", "")

tokernizer_output2 = tokenizer([sentence, sentence2], padding=True)
tokernizer_output2

# attention masks tell the model which tokens to ignore, such as padding
"""
{'input_ids': [[101, 1332, 1297, 3114, 1128, 22782, 1116, 117, 1274, 112, 189, 1294, 22782, 6397, 102], [101, 1332, 1297, 3114, 1128, 22782, 1116, 117, 1294, 22782, 6397, 102, 0, 0, 0]], 
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]}"""

# %%
