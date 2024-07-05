import torch
import torch.nn as nn

class TempleteLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x, y)