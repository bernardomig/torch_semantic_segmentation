from functools import partial
from torch.utils.data import DataLoader
from torch_semantic_segmentation.metrics import RunningMetric, AveragedMetric
from torch_semantic_segmentation.metrics import accuracy, miou, confusion_matrix
import os
import tempfile
from tqdm import tqdm

from mlflow import (log_metrics, log_params, log_artifacts, start_run, end_run)

import numpy as np
import torch
from torch import nn

from mlflow import (
    log_artifact, log_metric, log_param, log_metrics, log_params)

import argparse
import logging

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    help='the name of the model to train',
                    choices=['enet'], required=True)
parser.add_argument('--batch-size', type=int,
                    help='the batch size',
                    required=True)
parser.add_argument('--device', type=int,
                    help='the device to train the model on (for example, cuda:0)',
                    default=0)
parser.add_argument('--epochs', type=int,
                    help='the number of epochs to train the model',
                    required=True)
parser.add_argument('--lr', type=float,
                    help='the learning rate',
                    required=True)
parser.add_argument('--crop-size', type=int,
                    help='the crop size to train the model on (for example, --crop-size 512 256)',
                    required=True, nargs=2)
parser.add_argument('--scaling', type=float,
                    help='the limits of the range of scaling',
                    required=True, nargs=2)
parser.add_argument('--loss-fn', type=str,
                    help='the name of the loss function to train the model',
                    default='cross-entropy')
parser.add_argument('--evaluate-freq', type=int,
                    help='perform the evaluation every n epochs',
                    default=5)
args = parser.parse_args()

hparams = {
    'model': args.model,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'lr': args.lr,
    'crop_size': 'x'.join(map(str, args.crop_size)),
    'scaling': '-'.join(map(str, args.scaling)),
    'loss_fn': args.loss_fn,
}

device = torch.device(args.device)


def get_augmentation_fns(crop_size, scaling):
    from albumentations import (
        Compose, Normalize,
        RandomResizedCrop,
        HorizontalFlip,
        Rotate,
        Resize,
        RandomBrightnessContrast,
        RandomBrightness,
        RandomCrop,
        RandomScale,
        RandomSizedCrop,
        RandomGamma,
    )
    from albumentations.pytorch import ToTensorV2

    train_augmentations = Compose([
        RandomScale(scaling),
        RandomCrop(crop_size[1], crop_size[0]),
        HorizontalFlip(),
        Normalize(),
        ToTensorV2(),
    ])

    val_augmentations = Compose([
        Normalize(),
        ToTensorV2(),
    ])

    return train_augmentations, val_augmentations


def get_dataset(train_augmentations, val_augmentations):
    from torch_semantic_segmentation.data import CityScapesDataset

    train_ds = CityScapesDataset(
        '/home/bml/datasets/cities-scapes',
        split='train', transforms=train_augmentations)

    train_eval_ds = CityScapesDataset(
        '/home/bml/datasets/cities-scapes',
        split='train', transforms=val_augmentations)

    val_eval_ds = CityScapesDataset(
        '/home/bml/datasets/cities-scapes',
        split='val', transforms=val_augmentations)

    return train_ds, train_eval_ds, val_eval_ds


def get_loss_fn(name='cross-entropy'):
    from torch_semantic_segmentation.losses import ohem_loss, soft_dice_loss

    if name == 'cross-entropy':
        return nn.CrossEntropyLoss(ignore_index=255)
    elif name == 'ohem':
        return partial(ohem_loss, ignore_index=255)
    elif name == 'dice':
        return partial(soft_dice_loss, num_classes=19, ignore_index=255)
    elif name == 'balanced-cross-entropy':
        counts = torch.load('city-weights.pth')[:19]
        weight = 1. / torch.log(1.02 + counts)
        return nn.CrossEntropyLoss(ignore_index=255, weight=weight, reduction='mean')
    else:
        raise ValueError('unknown loss_fn')


def get_model(name='fastscnn'):
    from torch_semantic_segmentation.models import FastSCNN, ENet

    if name == 'fastscnn':
        return FastSCNN
    elif name == 'enet':
        return ENet
    else:
        raise ValueError('unknown model')

## BEGIN TRAINING ##


model = get_model(hparams['model'])(3, 19)
model = model.to(device)

# optimizer = torch.optim.SGD(
#     model.parameters(), lr=hparams['lr'], weight_decay=1e-5, momentum=0.9)

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9, nesterov=True)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, [50, 100, 150])

loss_fn = get_loss_fn(hparams['loss_fn'])
if isinstance(loss_fn, nn.Module):
    loss_fn = loss_fn.to(device)


criterion = loss_fn


def update_fn(batch):
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


state = {
    'epoch': 0,
    'iteration': 0,
    'loss': RunningMetric(lambda x: x),
}


# FOR MLFLOW LOGGING
start_run()

log_params(hparams)


def evaluate_fn(dataloader, metric_fns):
    metrics = {name: AveragedMetric(metric_fn)
               for name, metric_fn in metric_fns.items()}

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        for metric in metrics:
            metrics[metric].update(outputs, targets)
    return metrics


train_aug, val_aug = get_augmentation_fns(args.crop_size, args.scaling)

train_ds, train_eval_ds, val_eval_ds = get_dataset(
    train_aug, val_aug)

train_loader = DataLoader(train_ds, num_workers=8,
                          batch_size=hparams['batch_size'], shuffle=True)
train_eval_loader = DataLoader(train_eval_ds, num_workers=8, batch_size=5)
val_eval_loader = DataLoader(val_eval_ds, num_workers=8, batch_size=5)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=hparams['lr'],
    epochs=hparams['epochs'],
    steps_per_epoch=len(train_loader)
)

metric_fns = {
    'accuracy': partial(accuracy, ignore_index=255),
    'cm': partial(confusion_matrix, num_classes=19),
    'loss': (lambda x, y: loss_fn(x, y).mean()),
}

beta_dist = torch.distributions.Beta(0.1, 0.1)

for epoch in range(hparams['epochs']):
    state['epoch'] = epoch

    # first, train
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # MIXUP PART
        # get the beta
        lamb = beta_dist.sample().to(device)
        perm = torch.randperm(inputs.shape[0]).to(device)
        inputs1 = inputs
        inputs2 = inputs[perm]
        targets1 = targets
        targets2 = targets[perm]

        inputs = (lamb * inputs1 + (1 - lamb) * inputs2)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = (lamb * criterion(outputs, targets1) +
                (1 - lamb) * criterion(outputs, targets2))
        loss.backward()
        optimizer.step()

        state['loss'].update(loss.item())
        state['iteration'] += 1

        if state['iteration'] % 20 == 0:
            logging.info("Epoch {:03d}/{:03d} Iteration {:03d}: loss = {:03f}"
                         .format(epoch, hparams['epochs'], state['iteration'], state['loss'].compute()))

            log_metrics({
                'train/loss': loss.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                # 'train/mom': optimizer.param_groups[0]['momentum']
            }, step=state['iteration'])

        scheduler.step()

    if epoch % 5 == 0:
        metrics = evaluate_fn(val_eval_loader, metric_fns)
        logging.info("Evaluating for val (epoch {:03d}/{:03d}): loss = {}, miou = {}"
                     .format(epoch, hparams['epochs'], metrics['loss'].compute(), miou(metrics['cm'].compute())))
        log_metrics({
            'val/loss': metrics['loss'].compute().item(),
            'val/miou': miou(metrics['cm'].compute()).item(),
            'val/acc': metrics['accuracy'].compute().item(),
        }, step=state['iteration'])

        torch.save(model.state_dict(),
                   f'checkpoints-enet-mixup/enet-cityscapes-{epoch:03d}.pth')

end_run()
