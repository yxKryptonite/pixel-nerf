import torch

def pixel_loss(pred, gt):
    return torch.nn.functional.mse_loss(pred, gt)