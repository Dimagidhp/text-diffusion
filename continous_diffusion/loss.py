import torch
from torch import nn, Tensor
import einops

from .embedding import Embedder
from .scheduling import AdaptiveSchedule

class Loss(nn.Module):
    def __init__(self, noise_schedule: AdaptiveSchedule):
        super().__init__()
        self.noise_schedule = noise_schedule
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, target_tokens: Tensor, logits: Tensor, sigma: Tensor, attn_mask: Tensor) -> Tensor:
        # Transform the output embeddings back to logits
        logits = einops.rearrange(logits, 'b ... c -> b c (...)')

        # Flatten target tokens 
        target_tokens = target_tokens.flatten(start_dim=1)
        attn_mask = attn_mask.flatten(start_dim=1)

        # Compute cross-entropy loss
        loss = self.cross_entropy_loss(logits, target_tokens)
        
        # Apply mask to only compute loss on non-padding tokens
        loss = loss * attn_mask.float()
        
        # Compute average loss per sequence, avoiding division by zero
        valid_tokens_per_seq = attn_mask.float().sum(dim=-1)
        valid_tokens_per_seq = torch.clamp(valid_tokens_per_seq, min=1.0)  # Avoid division by zero
        loss_per_seq = loss.sum(dim=-1) / valid_tokens_per_seq

        # Check for NaN values and replace with 0
        loss_per_seq = torch.where(torch.isnan(loss_per_seq), torch.zeros_like(loss_per_seq), loss_per_seq)
        
        # Update the adaptive schedule with the current loss and sigma (useful for plotting)
        self.noise_schedule.add_data(loss_per_seq.detach(), sigma.detach())

        return loss_per_seq.mean()
