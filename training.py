# %%
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR

from continous_diffusion.diffusion import DiffusionModel
from continous_diffusion.callbacks import SchedulerUpdater, PlottingData, WriteText, FindUnused

from datasets import load_dataset
from transformers import AutoTokenizer
import composer
from composer.loggers import WandBLogger

import os

if __name__ == "__main__":
    #dataset = load_dataset("roneneldan/TinyStories")['train']
    # dataset_path = os.path.expanduser("~/.cache/huggingface/datasets/roneneldan___parquet/roneneldan--TinyStories-a62fc98e062666ca")
    # dataset = load_dataset(dataset_path)['train']
    # Load the WikiText-103 dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # gpt-2 tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"vocab_size: {tokenizer.vocab_size}")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")  

    # device="cuda" if torch.cuda.is_available() else "cpu"
    # %%
    #embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 1024, 1024, 8, 128, 8 #paper parameters (not sure about qkv_dim)  
    embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 256, 1024, 512, 8, 128, 8 
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 2048, 16, 128, 20  
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 1024, 16, 64, 8 
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 128, 512, 8, 64, 2  
    model=DiffusionModel(embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond=0.4,p_mask_cond=0.0,p_mask=0,prefix=0)

    print(f"n parameters:{model.n_parameters/1e6}M")
    # model.load_state_dict(torch.load('checkpoints/ep1_0.961538M'))
    # model=torch.compile(model)

    # %%
    # sampler=composer.utils.dist.get_sampler(tokenized_datasets['input_ids'])
    # train_loader = DataLoader(tokenized_datasets['input_ids'], batch_size=512, sampler=sampler)
    train_split = tokenized_datasets['train']
    sampler = composer.utils.dist.get_sampler(train_split)
    train_loader = DataLoader(train_split, batch_size=512, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4) # peak LR

    n_iters = 3000
    warmup_frac = 0.10
    warmup_iters = int(n_iters * warmup_frac)  # 300
    min_lr_start = 3e-6
    final_lr = 3e-5
    
    # start_factor scales from min_lr_start -> peak lr
    start_factor = min_lr_start / optimizer.param_groups[0]["lr"]  # 3e-6 / 3e-4 = 0.01
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_iters - warmup_iters, eta_min=final_lr)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

    callbacks=[PlottingData(200,model),SchedulerUpdater(200,model),WriteText(1000,model)]

    # Weights & Biases logger
    run_name = f"{model.n_parameters/1e6:.2f}M_{n_iters}it"
    wandb_logger = WandBLogger(project="text-diffusion-WikiText103", name=run_name, log_artifacts=True)


    #%%
    trainer=composer.Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=None,
        max_duration=f'{n_iters}it',
        device='gpu',
        callbacks=callbacks,
        loggers=[wandb_logger],
        run_name=run_name,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        save_folder="./checkpoints",
        save_filename="it{batch:06d}_of_3000_" + f"{model.n_parameters/1e6:.2f}M.pt",
        save_latest_filename="latest",
        save_overwrite=True,
        save_interval='1000it',
        algorithms=FindUnused() #necessary for self-conditioning when training with multi-gpu
    )

    trainer.fit()
                