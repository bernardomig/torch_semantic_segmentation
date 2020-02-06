from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage, ConfusionMatrix, Average, mIoU
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from torch_semantic_segmentation.losses import soft_dice_loss
from functools import partial
from torch_semantic_segmentation.data import CityScapesDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor
import os
import tempfile
import argparse

from torch_semantic_segmentation.metrics import accuracy
import numpy as np
import torch
from torch import nn

from torch_semantic_segmentation.models import ENet


from apex import amp

LOG_RESULTS = True

USE_WANDB = LOG_RESULTS
if USE_WANDB:
    import wandb
USE_MLFLOW = LOG_RESULTS
if USE_MLFLOW:
    from mlflow import (log_metrics, log_params,
                        log_artifacts, start_run, end_run)


def create_trainer(
        model, optimizer, loss_fn, device,
        dataloaders,
        scheduler=None, scheduler_update_policy='batch',
        use_f16=False,
        use_mixup=False):

    train_dataloader, val_dataloader = dataloaders

    if use_f16:
        model, optimizer = amp.initialize(model, optimizer)

    def prepare_batch(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def update_fn(trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if use_f16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss.item()

    def validate_fn(trainer, batch):
        model.eval()
        x, y = prepare_batch(batch)
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y

    trainer = Engine(update_fn)
    RunningAverage(output_transform=lambda x: x) \
        .attach(trainer, 'loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed(engine):
        print("epoch {} completed"
              .format(engine.state.epoch))

    if scheduler is not None:
        update_policy = (Events.ITERATION_COMPLETED
                         if scheduler_update_policy == 'batch'
                         else Events.EPOCH_COMPLETED)

        @trainer.on(update_policy)
        def update_scheduler(engine):
            scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_optimizer_params(engine):
        param_groups = optimizer.param_groups[0]
        for h in ['lr', 'momentum', 'weight_decay']:
            if h in param_groups.keys():
                engine.state.metrics[h] = param_groups[h]

    @trainer.on(Events.ITERATION_COMPLETED(every=20))
    def log_training(engine):
        iteration = engine.state.iteration
        metrics = engine.state.metrics
        print("Iteration {:04d}: {}"
              .format(iteration, metrics))

    def create_evaluator(name):
        evaluator = Engine(validate_fn)

        Loss(loss_fn).attach(evaluator, 'loss')
        mIoU(ConfusionMatrix(num_classes=19)).attach(evaluator, 'mIOU')
        Average(output_transform=lambda x: accuracy(x[0], x[1], ignore_index=255)) \
            .attach(evaluator, 'accuracy')

        @evaluator.on(Events.COMPLETED)
        def mlflow_log_training(engine):
            iteration = trainer.state.iteration
            metrics = engine.state.metrics
            if USE_MLFLOW:
                log_metrics({
                    '{}/{}'.format(name, metric): value
                    for metric, value in metrics.items()
                }, step=iteration)
            if USE_WANDB:
                wandb.log({
                    '{}/{}'.format(name, metric): value
                    for metric, value in metrics.items()
                }, step=iteration)

        return evaluator

    val_evaluator = create_evaluator('val')
    train_evaluator = create_evaluator('train')

    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def evaluate(engine):
        print("starting validation")
        state = train_evaluator.run(train_dataloader)
        print("evaluation on train done: {}"
              .format(state.metrics))
        state = val_evaluator.run(val_dataloader)
        print("evaluation on val done: {}"
              .format(state.metrics))

    checkpoints_dir = tempfile.mkdtemp(prefix='checkpoints')
    print("logging to {}".format(checkpoints_dir))
    checkpointer = Checkpoint(
        to_save={'model': model},
        save_handler=DiskSaver(checkpoints_dir, create_dir=True),
        n_saved=5,
        filename_prefix='model',
        score_function=lambda _: val_evaluator.state.metrics['mIOU'],
        score_name='mIOU',
        global_step_transform=global_step_from_engine(trainer))
    val_evaluator.add_event_handler(Events.COMPLETED, checkpointer)

    # mlflow logging
    @trainer.on(Events.ITERATION_COMPLETED(every=20))
    def mlflow_log_training(engine):
        iteration = engine.state.iteration
        metrics = engine.state.metrics
        if USE_MLFLOW:
            log_metrics({
                'training/{}'.format(name): value
                for name, value in metrics.items()
            }, step=iteration)
        if USE_WANDB:
            wandb.log({
                'training/{}'.format(name): value
                for name, value in metrics.items()
            }, step=iteration)

    @trainer.on(Events.COMPLETED)
    def save_checkpoints(engine):
        print('saving checkpoints')
        if USE_MLFLOW:
            log_artifacts(checkpoints_dir, 'checkpoints')
        if USE_WANDB:
            wandb.save(os.path.join(checkpoints_dir, '*.pth'))

    return trainer


def create_tfms(crop_size, scaling):
    train_tfms = albu.Compose([
        albu.RandomScale(scaling),
        albu.RandomCrop(crop_size[1], crop_size[0]),
        albu.HorizontalFlip(),
        albu.Normalize(),
        ToTensor(),
    ])
    val_tfms = albu.Compose([
        albu.Resize(512, 1024),
        albu.Normalize(),
        ToTensor(),
    ])
    return train_tfms, val_tfms


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='the name of the model to train',
                        choices=['enet'], required=True)
    parser.add_argument('--batch-size', type=int,
                        help='the batch size',
                        required=True)
    parser.add_argument('--epochs', type=int,
                        help='the number of epochs to train the model',
                        required=True)
    parser.add_argument('--lr', type=float,
                        help='the learning rate',
                        required=True)
    parser.add_argument('--crop-size', type=int,
                        help='the crop size to train the model (i.e. --crop-size 512 256)',
                        required=True, nargs=2)
    parser.add_argument('--scaling', type=float,
                        help='the scaling limits',
                        required=True, nargs=2)
    parser.add_argument('--loss-fn', type=str,
                        help='the loss function to train the model',
                        choices=['cross-entropy', 'balanced-cross-entropy',
                                 'soft-dice', 'ohem'],
                        default='cross-entropy')
    parser.add_argument('--optimizer', type=str,
                        help='the optimizer to use',
                        choices=['sgd', 'sgd-nesterov', 'adamw'],
                        default='sgd-nesterov')
    parser.add_argument('--scheduler', type=str,
                        choices=['step', 'poly',
                                 'one-cycle'],
                        help='the learning rate scheduler to use')
    parser.add_argument('--scheduler-args', type=str,
                        help='the arguments to the scheduler')
    parser.add_argument('--weight-decay', type=float,
                        help='the weight decay to use',
                        required=True)
    parser.add_argument('--use-f16', action='store_true',
                        help='use mixed precision training')
    parser.add_argument('--use-mixup', action='store_true',
                        help='use the mixup training technique')
    parser.add_argument('--mixup-alpha', type=float,
                        help='the alpha value for the mixup training')
    args = parser.parse_args()

    crop_size = args.crop_size
    assert crop_size[0] % 16 == 0 and crop_size[1] % 16 == 0
    scaling = args.scaling
    assert scaling[0] <= scaling[1]
    assert 1024 * scaling[0] >= min(crop_size)

    config = {
        'model': args.model,
        'dataset': 'cityscapes',
        'epochs': args.epochs,
        'batch-size': args.batch_size,
        'crop-size': crop_size,
        'scaling': scaling,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'scheduler-args': args.scheduler_args,
        'loss-fn': args.loss_fn,
        'lr': args.lr,
        'weight-decay': args.weight_decay,
        'use-f16': args.use_f16,
        'use-mixup': args.use_mixup,
        'mixup-alpha': args.mixup_alpha,
    }

    if config['use-mixup']:
        assert config['mixup-alpha'] is not None

    _config = {
        'model': config['model'],
        'dataset': config['dataset'],
        'crop-size': 'x'.join(map(str, crop_size)),
        'epochs': config['epochs'],
        'scaling': '-'.join(map(str, scaling)),
        'optimizer': config['optimizer'],
        'scheduler': (
            config['scheduler']
            if config['scheduler-args'] is None
            else '{}:{}'.format(config['scheduler'], config['scheduler-args'])),
        'lr': config['lr'],
        'loss-fn': config['loss-fn'],
        'weight-decay': config['weight-decay'],
        'use-mixup': 'yes:alpha={}'.format(config['mixup-alpha']) if config['use-mixup'] else 'no',
        'use-f16': 'yes' if config['use-f16'] else 'no',
        'batch-size': config['batch-size'],
    }

    from pprint import pprint
    print("starting with config"),
    pprint(_config)

    if USE_WANDB:
        wandb.init(project='semantic-segmentation', config=_config)

    if USE_MLFLOW:
        log_params(_config)

    device = torch.device('cuda')

    train_tfms, val_tfms = create_tfms(config['crop-size'], config['scaling'])

    train_ds = CityScapesDataset(
        '/home/bml/datasets/cities-scapes',
        split='train', transforms=train_tfms)
    val_ds = [
        CityScapesDataset(
            '/home/bml/datasets/cities-scapes',
            split='train', transforms=val_tfms),
        CityScapesDataset(
            '/home/bml/datasets/cities-scapes',
            split='val', transforms=val_tfms),
    ]

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['batch-size'], num_workers=16, drop_last=True, shuffle=True, pin_memory=True)

    val_loaders = [
        torch.utils.data.DataLoader(
            ds, batch_size=8, num_workers=8, drop_last=False)
        for ds in val_ds]

    # configuring the model
    if config['model'] == 'enet':
        Model = ENet
    else:
        raise ValueError('unknown model')

    model = Model(3, 19).to(device)

    if USE_WANDB:
        wandb.watch(model)

    # configuring the optimizer
    if config['optimizer'] == 'sgd':
        Optimizer = partial(torch.optim.SGD, momentum=0.85)
    elif config['optimizer'] == 'sgd-nesterov':
        Optimizer = partial(torch.optim.SGD, momentum=0.85, nesterov=True)
    elif config['optimizer'] == 'adamw':
        Optimizer = torch.optim.AdamW
    else:
        raise ValueError('unknown optimizer')

    optimizer = Optimizer(
        model.parameters(),
        lr=config['lr'], weight_decay=config['weight-decay'])

    if config['scheduler'] is None:
        Scheduler = None
        scheduler_update_policy = None
    elif config['scheduler'] == 'step':
        sargs = dict((value.split('=')
                      for value in config['scheduler-args'].split(',')))
        step_size = int(sargs['step-size'])
        gamma = float(sargs['gamma'])
        Scheduler = partial(torch.optim.lr_scheduler.StepLR,
                            step_size=step_size, gamma=gamma)
        scheduler_update_policy = 'epoch'
    elif config['scheduler'] == 'one-cycle':
        Scheduler = partial(torch.optim.lr_scheduler.OneCycleLR,
                            max_lr=config['lr'],
                            epochs=config['epochs'],
                            steps_per_epoch=len(train_loader))
        scheduler_update_policy = 'batch'
    elif config['scheduler'] == 'poly':
        sargs = dict((value.split('=')
                      for value in config['scheduler-args'].split(',')))
        gamma = float(sargs['gamma'])
        Scheduler = partial(torch.optim.lr_scheduler.LambdaLR,
                            lr_lambda=lambda epoch: 1 - (epoch / config['epochs'])**gamma)
        scheduler_update_policy = 'epoch'

    if Scheduler is not None:
        scheduler = Scheduler(optimizer)
    else:
        scheduler = None

    counts = torch.load('city-weights.pth')[:19]
    weight = 1. / torch.log(1.02 + counts)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255, weight=weight).to(device)

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        dataloaders=val_loaders,
        scheduler=scheduler, scheduler_update_policy=scheduler_update_policy,
        use_f16=False,
        use_mixup=False,
    )

    # import sys
    # sys.exit(0)

    trainer.run(data=train_loader, max_epochs=config['epochs'])
