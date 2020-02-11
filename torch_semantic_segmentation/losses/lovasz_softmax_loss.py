
import torch
from torch import nn
from torch.nn import functional as F


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


def lovasz_softmax_loss(input, target, num_classes, ignore_index=None):
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


class LovaszSoftmaxLoss(nn.Module):

    def __init__(self, num_classes, ignore_index=-100):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return lovasz_softmax_loss(input, target,
                                   num_classes=self.num_classes,
                                   ignore_index=self.ignore_index)
