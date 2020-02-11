import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['dice_loss', 'DiceLoss']


def dice_loss(input, target, num_classes, smooth=1.0, ignore_index=-100):
    r"""The (Soft) Dice Loss.
    """

    logits = torch.softmax(input, dim=1).flatten()
    target = target.flatten(1)

    mask = target != ignore_index & target > 0 & target < num_classes
    logits = logits.permute(0, 2, 1)[mask]
    target = target[mask]

    target = F.one_hot(target, num_classes=num_classes).float()
    intersection = torch.sum(logits * target, dim=0)
    union = torch.sum(logits, dim=0) + torch.sum(target, dim=0)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return torch.mean(1. - dice)


class DiceLoss(nn.Module):

    def __init__(self, num_classes, smooth=1.0, ignore_index=-100):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return dice_loss(input, target,
                         num_classes=self.num_classes,
                         smooth=self.smooth,
                         ignore_index=self.ignore_index)
