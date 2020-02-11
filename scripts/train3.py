import os
import tempfile
from functools import partial
import logging
from logging import info
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Loss
from ignite.metrics.confusion_matrix import cmAccuracy, ConfusionMatrix, mIoU, IoU, DiceCoefficient
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from apex import amp

import wandb


logging.basicConfig(level=logging.INFO)


def load_model(name):
    from torch_semantic_segmentation.models import ENet
    if name == 'enet':
        return ENet
    else:
        raise ValueError('no model registered as {}'.format(name))


def load_dataset(name):
    from torch_semantic_segmentation.data import CityScapesDataset

    if name == 'cityscapes':
        config = {
            'in_channels': 3,
            'ignore_index': 255,
            'num_classes': 19,
            'img_size': [2048, 1024],
            'val_size': [1024, 512],
            'class_freq': CityScapesDataset.CLASS_FREQ,
        }
        return CityScapesDataset, config
    else:
        raise ValueError('no dataset registered as {}'.format(name))


def load_optimizer(name, **kwargs):
    from torch import optim
    optimizers = {
        'SGD': optim.SGD,
        'SGDNesterov': partial(optim.SGD, nesterov=True),
        'AdamW': optim.AdamW,
    }

    if name not in optimizers.keys():
        raise ValueError('no optimizer registered as {}'.format(name))

    return partial(optimizers[name], **kwargs)


def load_loss_fn(name, class_freq, ignore_index, **kwargs):
    from torch_semantic_segmentation.losses import (
        DiceLoss, OHEMLoss, FocalLoss, LovaszSoftmaxLoss)
    # from kornia.losses import DiceLoss

    if name == 'cross_entropy':
        return partial(torch.nn.CrossEntropyLoss, ignore_index=ignore_index)
    elif name == 'ohem':
        return partial(OHEMLoss, ignore_index=ignore_index, numel_frac=0.05)
    elif name == 'dice_loss':
        return partial(DiceLoss, num_classes=19, ignore_index=255)
    elif name == 'focal_loss':
        return partial(FocalLoss, alpha=1.0, gamma=2.0, ignore_index=ignore_index)
    elif name == 'lovasz_softmax_loss':
        return partial(LovaszSoftmaxLoss, num_classes=19, ignore_index=255)
    else:
        raise ValueError("no loss fn registered as {}".format(name))


def create_tfms(crop_size, scaling, validation_size):
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2 as ToTensor
    train_tfms = albu.Compose([
        albu.RandomScale(scaling),
        albu.RandomCrop(crop_size[1], crop_size[0]),
        albu.HorizontalFlip(),
        albu.Normalize(),
        ToTensor(),
    ])
    val_tfms = albu.Compose([
        albu.Resize(validation_size[1], validation_size[0]),
        albu.Normalize(),
        ToTensor(),
    ])
    return train_tfms, val_tfms


def create_trainer(model, optimizer, loss_fn, device, use_f16=False, use_mixup=False, mixup_alpha=0.1):
    def prepare_batch(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def update_fn(_trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch)

        if use_mixup:
            batch_size = x.shape[0]
            beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            betas = beta_dist.sample((batch_size, 1, 1, 1))

            # shuffle x
            ind = torch.randperm(batch_size)
            x_ = x[ind]

            x = (betas * x) + ((1 - betas) * x_)

        y_pred = model(x)

        if use_mixup:
            # loss = loss_fn(y_pred, y)
            loss = betas * loss_fn(y_pred, y) + \
                (1 - betas) * loss_fn(y_pred, y[ind])
            loss = loss.mean()
        else:
            loss = loss_fn(y_pred, y)

        if use_f16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x, epoch_bound=False) \
        .attach(trainer, 'loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_optimizer_params(engine):
        param_groups = optimizer.param_groups[0]
        for h in ['lr', 'momentum', 'weight_decay']:
            if h in param_groups.keys():
                engine.state.metrics[h] = param_groups[h]

    @trainer.on(Events.ITERATION_COMPLETED(every=5))
    def log_loss(trainer):
        info("Iteration {:03d}: loss = {:.03f}"
             .format(trainer.state.iteration, trainer.state.metrics['loss']))

    return trainer


def create_evaluator(model, loss_fn, num_classes, device):
    def prepare_batch(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def validate_fn(_trainer, batch):
        model.eval()
        x, y = prepare_batch(batch)
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y

    evaluator = Engine(validate_fn)

    Loss(loss_fn).attach(evaluator, 'loss')
    IoU(ConfusionMatrix(num_classes)).attach(evaluator, 'IOU')
    mIoU(ConfusionMatrix(num_classes)).attach(evaluator, 'mIOU')
    cmAccuracy(ConfusionMatrix(num_classes)).attach(evaluator, 'accuracy')
    DiceCoefficient(ConfusionMatrix(num_classes)).attach(evaluator, 'dice')

    return evaluator


config = {
    'dataset': 'cityscapes',
    'model': 'enet',
    'optimizer': 'SGDNesterov',
    'loss': 'focal_loss',
    # 'loss.numel_frac': 0.05,
    'loss.gamma': 2.0,
    'loss.alpha': 1.0,

    'lr': 1e-2,
    'weight_decay': 1e-4,

    'scheduler': 'OneCycle',

    'epochs': 300,
    'batch_size': 16,
    'use_f16': False,

    'crop_size': [512, 512],
    'scaling_factors': [0.50, 1.5],
}

DEVICE_ID = 3
VAL_BATCH_SIZE = 4
USE_F16 = False
NUM_WORKERS = 8
NUM_WORKERS_VAL = 6
DATASET_DIR = '/home/bml/datasets/cities-scapes'
EVALUATE_FREQ = 5


wandb.init(project='semantic-segmentation', config=config)

torch.cuda.set_device(DEVICE_ID)
device = torch.device('cuda', DEVICE_ID)

Model = load_model(config['model'])
Dataset, ds_config = load_dataset(config['dataset'])
num_classes = ds_config['num_classes']

LossFn = load_loss_fn(
    config['loss'], ds_config['class_freq'], ignore_index=ds_config['ignore_index'])

Optimizer = load_optimizer(config['optimizer'], momentum=0.85)

train_tfms, val_tfms = create_tfms(
    crop_size=config['crop_size'],
    scaling=config['scaling_factors'],
    validation_size=ds_config['val_size'])

train_ds = Dataset(
    DATASET_DIR, split='train', transforms=train_tfms)
val_ds = OrderedDict([
    ('train', Dataset(DATASET_DIR, split='train', transforms=val_tfms)),
    ('val', Dataset(DATASET_DIR, split='val', transforms=val_tfms)),
])


train_loader = DataLoader(
    train_ds, batch_size=config['batch_size'], num_workers=NUM_WORKERS, pin_memory=True)
val_loaders = OrderedDict([
    (name, DataLoader(ds, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS_VAL))
    for name, ds in val_ds.items()
])


model = Model(in_channels=ds_config['in_channels'],
              out_channels=num_classes)
# model.load_state_dict(torch.load(
#     'torch_segmentation_models/different_loss_fns/enet_citys_loss=ohem_mIOU=0.4692pth', map_location='cpu'))
model = model.to(device)
optimizer = Optimizer(model.parameters(),
                      lr=config['lr'], weight_decay=config['weight_decay'])

wandb.watch(model)

if config['use_f16']:
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", keep_batchnorm_fp32=True)

loss_fn = LossFn()

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=config['lr'], epochs=config['epochs'], steps_per_epoch=len(train_loader))


trainer = create_trainer(model, optimizer, loss_fn,
                         device, use_f16=config['use_f16'])


@trainer.on(Events.ITERATION_COMPLETED)
def step_scheduler(_trainer):
    scheduler.step()


@trainer.on(Events.ITERATION_COMPLETED(every=EVALUATE_FREQ))
def log_training_metrics(engine):
    current_step = engine.state.iteration
    metrics = engine.state.metrics
    wandb.log({
        'training/{}'.format(metric): value
        for metric, value in metrics.items()
    }, step=current_step)


train_evaluator = create_evaluator(
    model, loss_fn, num_classes=num_classes, device=device)
val_evaluator = create_evaluator(
    model, loss_fn, num_classes=num_classes, device=device)


def log_metrics(evaluator, name):
    current_step = trainer.state.iteration
    metrics = evaluator.state.metrics
    wandb.log({
        '{}/{}'.format(name, metric): value
        for metric, value in metrics.items()
    }, step=current_step)


train_evaluator.add_event_handler(
    Events.COMPLETED, partial(log_metrics, name='train'))
val_evaluator.add_event_handler(
    Events.COMPLETED, partial(log_metrics, name='val'))


@trainer.on(Events.EPOCH_COMPLETED(every=EVALUATE_FREQ))
def evaluate(engine):
    def format_metrics(metrics):
        return ', ' \
            .join([f"{name}={value}"
                   for name, value in state.metrics.items()
                   if name in ['loss', 'mIOU', 'accuracy']])

    info("starting evaluation")
    state = train_evaluator.run(val_loaders['train'])
    info("evaluation on train done: {}"
         .format(format_metrics(state.metrics)))
    state = val_evaluator.run(val_loaders['val'])
    info("evaluation on val done: {}"
         .format(format_metrics(state.metrics)))
    info("finished evaluation")


checkpoints_dir = os.path.join(wandb.run.dir, 'checkpoints')
info("logging checkpoints to {}".format(checkpoints_dir))
checkpointer = Checkpoint(
    to_save={'model': model},
    save_handler=DiskSaver(checkpoints_dir, create_dir=True),
    n_saved=5,
    filename_prefix='model',
    score_function=lambda _: val_evaluator.state.metrics['mIOU'],
    score_name='mIOU',
    global_step_transform=global_step_from_engine(trainer))
val_evaluator.add_event_handler(Events.COMPLETED, checkpointer)

trainer.run(train_loader, max_epochs=config['epochs'])
