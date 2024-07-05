import torch
from torch.utils.data import Dataset,DataLoader

class RandomDataset(Dataset):
    def __init__(self):
        self.len = 200000

    def __getitem__(self, index):
        x = torch.randn(4, 101,64)
        y = torch.randn(4, 101,64)
        return x,y

    def __len__(self):
        return self.len