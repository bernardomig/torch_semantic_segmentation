import tempfile
import os

import numpy as np
import torch
from torch import nn

from tqdm.auto import tqdm

import argparse

import logging

from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from ignite.metrics import (
    Accuracy, Loss, IoU, ConfusionMatrix, RunningAverage, mIoU)
from ignite.engine import (
    Events, create_supervised_trainer, create_supervised_evaluator)
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, OneCycleLR
from ignite.contrib.handlers import LRScheduler

from mlflow import (
    log_artifact, log_metric, log_param, log_metrics, log_params, set_tags)

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

from torch_semantic_segmentation.data import DeepDriveDataset


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--batch-size', required=True, type=int)
parser.add_argument('--device', required=True, type=str)
parser.add_argument('--epochs', required=True, type=int)
parser.add_argument('--learning-rate', required=True, type=float)

args = parser.parse_args()

log_params({
    'model': args.model,
    'learning-rate': args.learning_rate,
    'batch-size': args.batch_size,
    'epochs': args.epochs,
})

device = torch.device(args.device)

# logging.basicConfig(level=logging.DEBUG)
logging.info(
    f"Starting with arguments: model={args.model}, batch-size={args.batch_size},"
    f" epochs={args.epochs}, learning-rate={args.learning_rate}, device={args.device}"
)

# data-augmentation
train_augmentations = Compose([
    RandomScale((0.75, 1.5)),
    RandomCrop(512, 512),
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
train_ds = DeepDriveDataset('/home/bml/datasets/bdd100k/seg',
                            split='train', transforms=train_augmentations)
val_ds = DeepDriveDataset('/home/bml/datasets/bdd100k/seg',
                          split='val', transforms=val_augmentations)
# data loading
train_loader = DataLoader(train_ds, num_workers=4, batch_size=args.batch_size)
val_loader = DataLoader(val_ds, num_workers=4, batch_size=args.batch_size)


criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
model = ENet(3, 19)


max_epochs = int(args.epochs)

optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-5)


lr_scheduler = OneCycleLR(
    optimizer=optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_ds),
    epochs=max_epochs,
)


trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

ProgressBar(persist=True).attach(
    trainer, ['loss'])

scheduler = LRScheduler(lr_scheduler)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
    'mIOU': mIoU(ConfusionMatrix(num_classes=19)),
}


evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
ProgressBar(persist=False).attach(evaluator)


checkpoints_dir = tempfile.mkdtemp()


@trainer.on(Events.EPOCH_COMPLETED)
def validate(trainer):
    state = evaluator.run(train_loader)
    logging.info(f"train metrics: {state.metrics}")
    log_metrics({
        'train/loss': state.metrics['loss'],
        'train/acc': state.metrics['accuracy'],
        'train/mIOU': state.metrics['mIOU'],
    })

    state = evaluator.run(val_loader)
    logging.info(f"evaluation metrics: {state.metrics}")
    log_metrics({
        'val/loss': state.metrics['loss'],
        'val/acc': state.metrics['accuracy'],
        'val/mIOU': state.metrics['mIOU'],
    })

    epoch = trainer.state.epoch
    mIOU = state.metrics['mIOU']
    checkpoint_file = os.path.join(
        checkpoints_dir, f'{args.model}-{epoch:03d}-{mIOU}.pth')
    torch.save(model.state_dict(), checkpoint_file)

    logging.info(f"saving model to {checkpoint_file}")


trainer.run(train_loader, max_epochs=args.epochs)

log_artifact(checkpoints_dir, 'checkpoints')
