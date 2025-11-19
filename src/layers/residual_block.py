import torch
import torch.nn as nn
from .conv_layer import ConvLayer
from .shortcut_layer import Shortcut
from .shakedrop_layer import ShakeDrop

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.5):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = Shortcut(in_channels, out_channels, stride)
        self.shakedrop = ShakeDrop(p_drop=p_drop)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.shakedrop(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
