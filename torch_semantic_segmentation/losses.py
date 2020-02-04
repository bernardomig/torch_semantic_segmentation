
from math import log
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['focal_loss', 'FocalLoss',
           'ohem_loss', 'OHEMLoss',
           'soft_dice_loss', 'SoftDiceLoss',
           'lovasz_softmax']


def focal_loss(input, target, gamma, ignore_index=None):
    "The focal loss"
    scores = F.softmax(input, dim=1)
    factor = torch.pow(1. - scores, gamma)
    log_score = factor * F.log_softmax(input, dim=1)
    return F.nll_loss(log_score, target, ignore_index=ignore_index)


class FocalLoss(nn.Module):

    def __init__(self, gamma, ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target,
                          gamma=self.gamma,
                          ignore_index=self.ignore_index)


def ohem_loss(input, target, ignore_index=None, thresh_loss=-log(0.7), min_numel_frac=0.16):
    loss = F.cross_entropy(input, target, ignore_index=255, reduction='none')
    loss = loss.flatten()
    n = int(loss.numel() * min_numel_frac)

    loss, _ = torch.sort(loss, descending=True)

    if loss[n] > thresh_loss:
        return loss[loss > 0.7].mean()
    else:
        return loss[:n].mean()


class OHEMLoss(nn.Module):

    def __init__(self, ignore_index=None, thresh_loss=-log(0.7), min_numel_frac=0.16):
        super().__init__()

        self.ignore_index = ignore_index
        self.thresh_loss = thresh_loss
        self.min_numel_frac = min_numel_frac

    def forward(self, input, target):
        return ohem_loss(input, target,
                         ignore_index=self.ignore_index,
                         thresh_loss=self.thresh_loss,
                         min_numel_frac=self.min_numel_frac)


def soft_dice_loss(inputs, target, num_classes, ignore_index=None):
    logits = torch.softmax(inputs, dim=1).flatten(2)
    target = target.flatten(1)

    if ignore_index is not None:
        mask = target != ignore_index
        logits = logits.permute(0, 2, 1)[mask]
        target = target[mask]

    target = F.one_hot(target, num_classes=num_classes).float()

    intersection = torch.sum(logits * target, dim=0)
    union = torch.sum(logits, dim=0) + torch.sum(target, dim=0)

    return 1. - (2. * intersection / union).mean()


class SoftDiceLoss(nn.Module):

    def __init__(self, num_classes, ignore_index=None):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return soft_dice_loss(input, target,
                              num_classes=self.num_classes,
                              ignore_index=self.ignore_index)


def lovasz_grad(gt_sorted):
    "Computes the gradient of the Lovasz extenstion w.r.t. sorted errors"
    p = len(gt_sorted)
    gts = torch.sum(gt_sorted)

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1. - gt_sorted).float().cumsum(0)
    jaccard = 1. - (intersection / union)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:1]
    return jaccard


def lovasz_softmax(input, target, num_classes, ignore_index=None):
    input = F.softmax(input, dim=1)

    # flatten the
    input = input.permute(0, 2, 3, 1).flatten(0, 2)
    target = target.flatten()

    # remove the ignored indices
    if ignore_index is not None:
        mask = target != ignore_index
        input = input[mask]
        target = target[mask]

    losses = []
    for c in range(num_classes):
        fg = (target == c).float()
        if fg.sum() == 0:
            continue
        pred = input[:, c]
        errors = (fg - pred).abs()
        indices = torch.argsort(errors, dim=0, descending=True)
        errors = errors[indices]
        fg = fg[indices]
        loss = torch.dot(errors, lovasz_grad(fg))
        losses.append(loss)
    return torch.stack(losses).mean()
