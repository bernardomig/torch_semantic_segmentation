from math import log
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['ohem_loss', 'OHEMLoss']


def ohem_loss(input, target, ignore_index=-100, thresh_loss=-log(0.7), numel_frac=0.01):
    loss = F.cross_entropy(
        input, target, ignore_index=ignore_index, reduction='none')
    loss = loss.flatten()
    n = int(loss.numel() * numel_frac)

    loss, _ = torch.sort(loss, descending=True)

    if loss[n] > thresh_loss:
        return loss[loss > thresh_loss].mean()
    else:
        return loss[:n].mean()


class OHEMLoss(nn.Module):

    def __init__(self, ignore_index=-100, thresh_loss=-log(0.7), numel_frac=0.01):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh_loss = thresh_loss
        self.numel_frac = numel_frac

    def forward(self, input, target):
        return ohem_loss(input, target,
                         ignore_index=self.ignore_index,
                         thresh_loss=self.thresh_loss,
                         numel_frac=self.numel_frac)
