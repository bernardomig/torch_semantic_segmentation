from torch_semantic_segmentation.losses import ohem_loss
from torch.optim.lr_scheduler import OneCycleLR
import tempfile
import os

import numpy as np
import torch
from torch import nn

import argparse

import logging

from mlflow import (
    log_artifact, log_metric, log_param, log_metrics, log_params)

from torch.utils.data import DataLoader

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

from torch_semantic_segmentation.models import ENet, FastSCNN

from torch_semantic_segmentation.data import DeepDriveDataset, CityScapesDataset

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--device', required=True, type=str)
parser.add_argument('--epochs', required=True, type=int)
parser.add_argument('--learning-rate', required=True, type=float)
parser.add_argument('--crop-size')

args = parser.parse_args()

crop_size = 256

log_params({
    'model': args.model,
    'learning-rate': args.learning_rate,
    'batch-size': args.batch_size,
    'epochs': args.epochs,
    'crop_size': crop_size,
})

device = torch.device(args.device)

# logging.basicConfig(level=logging.DEBUG)
logging.info(
    f"Starting with arguments: model={args.model}, batch-size={args.batch_size},"
    f" epochs={args.epochs}, learning-rate={args.learning_rate}, device={args.device}"
    f" crop-size={args.crop_size}"
)

# data-augmentation
train_augmentations = Compose([
    RandomScale((0.25, 1)),
    RandomCrop(256, 512),
    HorizontalFlip(),
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    # RandomGamma(),
    Normalize(),
    ToTensorV2(),
])

val_augmentations = Compose([
    Resize(512, 1024),
    Normalize(),
    ToTensorV2(),
])


# preparing dataset
# train_ds = DeepDriveDataset('/home/bml/datasets/bdd100k/seg',
#                             split='train', transforms=train_augmentations)
# val_ds = DeepDriveDataset('/home/bml/datasets/bdd100k/seg',
#                           split='val', transforms=val_augmentations))
train_ds = CityScapesDataset(
    '/home/bml/datasets/cities-scapes',
    split='train', transforms=train_augmentations)
logging.info("Training set len = {}".format(len(train_ds)))

train_eval_ds = CityScapesDataset(
    '/home/bml/datasets/cities-scapes',
    split='train', transforms=val_augmentations)

val_ds = CityScapesDataset(
    '/home/bml/datasets/cities-scapes',
    split='val', transforms=val_augmentations)

# data loading
train_loader = DataLoader(train_ds, num_workers=8,
                          batch_size=args.batch_size, shuffle=True)
train_eval_loader = DataLoader(train_eval_ds, num_workers=8,
                               batch_size=args.batch_size)
val_loader = DataLoader(val_ds, num_workers=4, batch_size=args.batch_size)


ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)


def criterion(input, target):
    y0, y1, y2 = input
    return ce_loss(y0, target) + 0.4 * ce_loss(y1, target) + 0.4 * ce_loss(y2, target)


model = FastSCNN(3, 19)


max_epochs = int(args.epochs)

optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-12, momentum=0.9, weight_decay=1e-5)


lr_scheduler = OneCycleLR(
    optimizer=optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=max_epochs,
)


def calculate_accuracy(input, target, ignore_index=255):
    labels = torch.argmax(input, dim=1)

    labels = labels[target != ignore_index]
    target = target[target != ignore_index]

    return (labels == target).float().mean()


state = {
    'epoch': 0,
    'iteration': 0,
}

model = model.to(device)


def evaluate(dataloader):
    logging.info("evaluating")

    model.eval()

    iterations = 0

    state = dict(accuracy=0, mIOU=0)
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_ = model(x)

        accuracy = calculate_accuracy(y_, y).item()
        cm = calculate_confusion_matrix(y_, y)
        mIOU = calculate_mIOU(cm).item()

        state['accuracy'] += accuracy
        state['mIOU'] += mIOU
        iterations += 1

    return {
        key: metric / iterations
        for key, metric in state.items()
    }


def training():
    logging.info("starting epoch {}".format(state['epoch']))

    model.train()

    loss_average = RunningAverage(criterion)
    accuracy = RunningAverage(calculate_accuracy)
    mIOU = RunningAverage()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_ = model(x)

        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_average.update(loss.item())
        accuracy.update(calculate_accuracy(y_[0], y).item())
        cm = calculate_confusion_matrix(y_[0], y)
        mIOU.update(calculate_mIOU(cm).item())

        if state['iteration'] % 20 == 0:
            logging.info('Iteration {:03d}: loss = {:.03f}, accuracy = {:.3f}, mIOU = {:.3f}'
                         .format(state['iteration'], loss_average.value, accuracy.value, mIOU.value))

            log_metrics({
                'training/loss': loss_average.value,
                'training/accuracy': accuracy.value,
                'training/mIOU': mIOU.value,
                'training/lr': optimizer.param_groups[0]['lr'],
                'training/momentum': optimizer.param_groups[0]['momentum'],
            }, step=state['iteration'])

        state['iteration'] += 1

    state['epoch'] += 1


for epoch in range(max_epochs):
    training()

    if epoch % 5 == 0:
        train_metrics = evaluate(train_eval_loader)
        val_metrics = evaluate(val_loader)

        results = {
            'acc/train': train_metrics['accuracy'],
            'acc/val': val_metrics['accuracy'],
            'mIOU/train': train_metrics['mIOU'],
            'mIOU/val': val_metrics['mIOU'],
        }
        log_metrics(results, step=epoch)

        logging.info("metrics: {}".format(results))
