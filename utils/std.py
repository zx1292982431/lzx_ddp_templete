import torch

def mean_std_normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized = (tensor - mean) / std
    return normalized