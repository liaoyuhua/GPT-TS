import torch
import torch.nn.functional as F


def mae_loss(prediction, target):
    """
    Args:
        prediction: shape (B, N, Y)
        target: shape (B, N, Y)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    loss = F.l1_loss(masked_prediction, masked_target)
    return loss


def mse_loss(prediction, target):
    """
    Args:
        prediction: shape (B, N, Y)
        target: shape (B, N, Y)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    loss = F.mse_loss(masked_prediction, masked_target)
    return loss
