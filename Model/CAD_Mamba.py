import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from Model.VSS_Block import MEVSSBlock

class CADmamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)
        gc=dim//2
        self.dw = nn.Conv2d(gc, gc, kernel_size=3, groups=gc, padding="same")
        self.block = MEVSSBlock(hidden_dim = gc)
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        skip = x
        x1, x2 = torch.chunk(x,2, dim=1)
        x1 = self.dw(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.block(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = self.dw(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.block(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x = torch.cat([x1, x2], dim=1)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        x= x * att_score  # Broadcasting to match the dimensions
        x= skip*self.scale+x
        return x