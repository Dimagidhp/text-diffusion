import torch
from torch import nn, Tensor


class MaskMaker(nn.Module):
    def __init__(self,masking_ratio,avg_prefix):
        super().__init__()
        self.masking_ratio=masking_ratio
        self.avg_prefix=avg_prefix
        
    def forward(self,shape,device='cpu'):
        mask=torch.rand(shape,device=device)<self.masking_ratio
        mask[:,:self.avg_prefix]=False
        return mask
