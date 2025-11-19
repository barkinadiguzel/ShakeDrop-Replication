import torch
import torch.nn as nn
from ..layers.residual_block import ResidualBlock
from ..layers.pool_layers.avgpool_layer import AvgPoolLayer
from ..layers.flatten_layer import FlattenLayer
from ..layers.fc_layer import FCLayer

class ResNetShakeDrop(nn.Module):
    def __init__(self, num_classes=10, layers=[2,2,2], channels=[16,32,64], p_drops=[0.5,0.5,0.5]):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = channels[0]
        for i, num_blocks in enumerate(layers):
            out_channels = channels[i]
            for j in range(num_blocks):
                stride = 2 if j==0 and i>0 else 1
                self.blocks.append(ResidualBlock(in_channels, out_channels, stride=stride, p_drop=p_drops[i]))
                in_channels = out_channels
        self.avgpool = AvgPoolLayer()
        self.flatten = FlattenLayer()
        self.fc = FCLayer(channels[-1], num_classes)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
