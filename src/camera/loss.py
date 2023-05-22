import torch

def pixel_loss(pred, gt):
    return torch.mean((pred - gt)**2)