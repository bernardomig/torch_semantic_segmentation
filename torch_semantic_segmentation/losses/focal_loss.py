import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['focal_loss', 'FocalLoss']


def focal_loss(input, target, alpha=0.25, gamma=2.0, ignore_index=-100):
    r"""The focal loss.
    """
    scores = F.softmax(input, dim=1)
    factor = torch.pow(1. - scores, gamma)
    return alpha * F.nll_loss(torch.exp(factor) * F.log_softmax(input, dim=1), target,
                              ignore_index=ignore_index)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target,
                          alpha=self.alpha,
                          gamma=self.gamma,
                          ignore_index=self.ignore_index)
