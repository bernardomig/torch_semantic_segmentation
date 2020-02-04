import torch


class RunningMetric:

    def __init__(self, metric_fn, alpha=0.1):
        self.metric_fn = metric_fn
        self.alpha = alpha

        self._value = None
        self._count = 0

    def update(self, *args, **kwargs):
        value = self.metric_fn(*args, **kwargs)
        if self._value is None:
            self._value = value
        else:
            self._value = value + (1. - self.alpha) * self._value
        self._count = 1. + (1. - self.alpha) * self._count

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "running average value can not be computed from no examples")
        return self._value / self._count


class AveragedMetric:

    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

        self._value = None
        self._count = 0

    def update(self, *args, **kwargs):
        value = self.metric_fn(*args, **kwargs)
        if self._value is None:
            self._value = value
        else:
            self._value = value + self._value
        self._count = 1 + self._count

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "average value can not be computed from no examples")
        return self._value / self._count


def accuracy(input, target, ignore_index=None):
    "Computes the accuracy"
    input = torch.argmax(input, dim=1)

    if ignore_index is not None:
        mask = target != ignore_index
        input = input[mask]
        target = target[mask]

    return (input == target).float().mean()


def error(input, target):
    "Computes the error rate (miss classified)"
    return 1.0 - accuracy(input, target)


def confusion_matrix(inputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    inputs = torch.argmax(inputs, dim=1).flatten()
    targets = targets.flatten()

    mask = (targets >= 0) & (targets < num_classes)
    inputs = inputs[mask]
    targets = targets[mask]

    indices = num_classes * targets + inputs
    m = torch \
        .bincount(indices, minlength=num_classes**2) \
        .reshape(num_classes, num_classes)
    return m.float() / targets.numel()


def iou(cm: torch.Tensor):
    return (torch.diag(cm) /
            (cm.sum(dim=1) + cm.sum(dim=0) - torch.diag(cm) + 1e-15))


def miou(cm: torch.Tensor):
    return iou(cm).mean()
