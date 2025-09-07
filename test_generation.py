# %%
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from continous_diffusion import DiffusionModel

# Load the TinyStories dataset
# dataset = load_dataset("roneneldan/TinyStories")
# Load the WikiText-103 dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use BERT tokenizer
print(f"vocab_size: {tokenizer.vocab_size}")
# Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")  

device="cuda" if torch.cuda.is_available() else "cpu"

# %%
embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 256, 896, 512, 8, 128, 10 

model=DiffusionModel(embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond=0.6,p_mask_cond=0.0,p_mask=0,prefix=0)


# Assuming state_dict is the loaded state dictionary and the prefix to remove is "_orig_mod."
state_dict=torch.load('checkpoints/139.96M_parameters_ep4.pt')
adjusted_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

# Now try loading the adjusted state dictionary
model.load_state_dict(adjusted_state_dict)

print(f"n parameters:{model.n_parameters/1e6}M")


# %%
out_embeddings=model.generate(1,128,1000,device=device)

from torch.distributions import Categorical

logits=model.un_embedder(out_embeddings)

distrubution=Categorical(logits=logits)
sample=distrubution.sample()

# sample=logits.argmax(dim=-1)

tokenizer.batch_decode(sample)
# %%
