import torch
import torch.nn.functional as F


def focal_bce_with_logits(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    focal = (1 - pt) ** gamma
    return (focal * bce).mean()
