import torch
import torch.nn as nn

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.shortcut(x)
