import torch
import torch.nn as nn

class ShakeDrop(nn.Module):
    def __init__(self, p_drop, alpha_range=(-1,1), beta_range=(0,1)):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        
    def forward(self, x):
        if not self.training:
            return x
        device = x.device
        gate = torch.rand(1, device=device).item()
        if gate < self.p_drop:
            alpha = torch.empty(1, device=device).uniform_(*self.alpha_range).item()
            beta = torch.empty(1, device=device).uniform_(*self.beta_range).item()
            return beta * x + alpha * x.detach()
        else:
            return x
