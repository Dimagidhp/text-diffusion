# %%
"""
Text Diffusion Training Script - Optimized Version

Key improvements implemented:
1. ✅ Pad tokens excluded from loss (already implemented in loss.py)
2. ✅ Lower peak LR for 148M params: 1e-4 (down from 2e-4)  
3. ✅ Optimized AdamW settings: betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
4. ✅ Cosine decay to 10% of base LR by the end
5. ✅ Mixed precision training with bf16 for better numerical stability
6. ✅ Gradient clipping to prevent exploding gradients
"""
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR

from continous_diffusion.diffusion import DiffusionModel
from continous_diffusion.callbacks import SchedulerUpdater, PlottingData, WriteText, FindUnused, LRMonitor

from datasets import load_dataset
from transformers import AutoTokenizer
import composer
from composer.loggers import WandBLogger
from composer.algorithms import GradientClipping

import os

if __name__ == "__main__":
    #dataset = load_dataset("roneneldan/TinyStories")['train']
    # dataset_path = os.path.expanduser("~/.cache/huggingface/datasets/roneneldan___parquet/roneneldan--TinyStories-a62fc98e062666ca")
    # dataset = load_dataset(dataset_path)['train']
    # Load the WikiText-103 dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Use BERT tokenizer for better CLS/SEP token handling
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")  

    # device="cuda" if torch.cuda.is_available() else "cpu"
    # %%
    #embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 1024, 1024, 8, 128, 8 #paper parameters (not sure about qkv_dim)  
    #embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 256, 1024, 512, 8, 128, 8 
    embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 256, 1024, 512, 8, 128, 8 
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 2048, 16, 128, 20  
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 256, 1024, 16, 64, 8 
    # embed_dim, hidden_dim, qkv_dim, num_heads, cond_dim, n_blocks = 64, 128, 512, 8, 64, 2  
    model=DiffusionModel(embed_dim,hidden_dim,qkv_dim,num_heads,cond_dim,n_blocks,tokenizer,p_self_cond=0.6,p_mask_cond=0.0,p_mask=0,prefix=0)

    print(f"n parameters:{model.n_parameters/1e6}M")
    # model.load_state_dict(torch.load('checkpoints/ep1_0.961538M'))
    # model=torch.compile(model)

    # %%
    # Create train and validation splits
    train_split = tokenized_datasets['train']
    val_split = tokenized_datasets['validation']
    
    # Create data loaders
    train_sampler = composer.utils.dist.get_sampler(train_split)
    train_loader = DataLoader(train_split, batch_size=64, sampler=train_sampler)
    
    # Create validation loader (no sampler needed for validation)
    val_loader = DataLoader(val_split, batch_size=64, shuffle=False)

    # Optimized AdamW settings for diffusion LMs (~148M params)
    base_lr = 1e-4  # Lower peak LR for 148M params as suggested
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),  # Recommended betas for diffusion LMs
        eps=1e-8,
        weight_decay=0.1    # Weight decay for regularization
    )
    iters_per_epoch = len(train_loader)
    print(f"iters per epoch: {iters_per_epoch}")

    n_iters = 20000
    warmup_frac = 0.1              
    warmup_iters = int(n_iters * warmup_frac)
    min_lr_start = 1e-6  # Start from very low LR
    final_lr = base_lr * 0.1  # Cosine decay to 10% of base LR 
    
    # start_factor scales from min_lr_start -> peak lr
    start_factor = min_lr_start / optimizer.param_groups[0]["lr"]  # 1e-6 / 1e-4 = 0.01
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_iters - warmup_iters, eta_min=final_lr)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

    callbacks=[PlottingData(200,model),SchedulerUpdater(200,model),WriteText(1000,model),LRMonitor(50)]

    # Weights & Biases logger
    run_name = f"{model.n_parameters/1e6:.2f}M_{n_iters}it_optimized"
    wandb_logger = WandBLogger(project="text-diffusion-WikiText103", name=run_name, log_artifacts=False)


    # Check hardware support for mixed precision
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    precision = 'amp_bf16' if bf16_supported else 'amp_fp16'
    print(f"Using mixed precision: {precision}")

    #%%
    trainer=composer.Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        eval_interval="500ba",  # Run validation every 500 batches
        max_duration=f'{n_iters}ba',
        device='gpu',
        precision=precision,  # Use bf16 if supported, fallback to fp16
        callbacks=callbacks,
        loggers=[wandb_logger],
        run_name=run_name,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        save_folder="./checkpoints",
        save_filename="it{batch:06d}_of_5000_" + f"{model.n_parameters/1e6:.2f}M.pt",
        save_latest_filename="latest",
        save_overwrite=True,
        save_num_checkpoints_to_keep=3,
        save_interval='2000ba',
        save_weights_only=True,  # Save only weights to avoid optimizer state issues
        algorithms=[FindUnused(), GradientClipping(clipping_type='norm', clipping_threshold=1.0)] #necessary for self-conditioning when training with multi-gpu
    )

    trainer.fit()
                