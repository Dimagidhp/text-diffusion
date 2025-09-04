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

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or any suitable tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size=tokenizer.vocab_size+1
print(f"vocab_size: {tokenizer.vocab_size}")
# Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")  

device="cuda" if torch.cuda.is_available() else "cpu"

# %%
model=DiffusionModel(embed_dim=256,
                     qkv_dim=4096,
                     num_heads=8,
                     cond_dim=64,
                     n_blocks=32,
                     vocab_size=vocab_size,
                     device=device
                     )


# Assuming state_dict is the loaded state dictionary and the prefix to remove is "_orig_mod."
state_dict=torch.load('checkpoints/139.96M_parameters_ep4.pt')
adjusted_state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

# Now try loading the adjusted state dictionary
model.load_state_dict(adjusted_state_dict)

print(f"n parameters:{model.n_parameters/1e6}M")


# %%
out_embeddings=model.generate(1,128,1000,device=device)

from torch.distributions.categorical import Categorical

logits=model.un_embedder(out_embeddings)

distrubution=Categorical(logits=logits)
sample=distrubution.sample()

# sample=logits.argmax(dim=-1)

tokenizer.batch_decode(sample)
# %%
