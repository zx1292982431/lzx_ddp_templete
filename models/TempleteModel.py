import torch
import torch.nn as nn

class TempleteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(64, 64, 3, padding=1)

    def forward(self, x):
        return self.conv1d(x)