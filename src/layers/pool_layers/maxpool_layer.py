import torch.nn as nn

class MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.pool(x)
