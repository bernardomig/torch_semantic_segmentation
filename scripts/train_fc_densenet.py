import argparse
import os
import tempfile

import logging
from logging import info


import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model


from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Loss
from ignite.metrics.confusion_matrix import cmAccuracy, ConfusionMatrix, mIoU, IoU, DiceCoefficient
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from torch_semantic_segmentation.models.fc_densenet import (
    fc_densenet_57, fc_densenet_67, fc_densenet_103)
from torch_semantic_segmentation.data import CityScapesDataset, DeepDriveDataset
from torch_semantic_segmentation.losses import OHEMLoss


logging.basicConfig(level=logging.INFO)

DATASET_DIR = '/srv/datasets/cityscapes'
MODELS = {'fc_densenet_57': fc_densenet_57,
          'fc_densenet_67': fc_densenet_67,
          'fc_densenet_103': fc_densenet_103}


def create_trainer(model, optimizer, loss_fn, device):
    from ignite.engine import _prepare_batch

    def update_fn(_trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
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

    if logging:
        @trainer.on(Events.ITERATION_COMPLETED(every=5))
        def log_loss(trainer):
            info("Iteration {:03d}: loss = {:.03f}, lr = {:04f}"
                 .format(trainer.state.iteration, trainer.state.metrics['loss'], trainer.state.metrics['lr']))

    return trainer


def create_evaluator(model, loss_fn, num_classes, device):
    from ignite.engine import _prepare_batch

    def validate_fn(_trainer, batch):
        model.eval()
        x, y = _prepare_batch(batch)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, required=True)

    parser.add_argument('--model', required=True,
                        choices=MODELS.keys())
    parser.add_argument('--state_dict', type=str, required=False)

    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()

    if args.distributed:
        dist.init_process_group('nccl', init_method='env://')
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        local_rank = args.local_rank
    else:
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')

    train_tfms = albu.Compose([
        albu.RandomScale([0.5, 2.0]),
        albu.RandomCrop(512, 768),
        albu.HorizontalFlip(),
        albu.Normalize(),
        ToTensor(),
    ])
    val_tfms = albu.Compose([
        albu.Normalize(),
        ToTensor(),
    ])

    train_dataset = CityScapesDataset(
        DATASET_DIR, split='train', transforms=train_tfms)
    val_dataset = CityScapesDataset(
        DATASET_DIR, split='val', transforms=val_tfms)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        kwargs = dict(num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(train_dataset, **kwargs)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, **kwargs)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        drop_last=True,
        num_workers=8,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        sampler=val_sampler,
    )

    model = MODELS[args.model](3, 19)
    if args.state_dict is not None:
        state_dict = torch.load(args.state_dict, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn = OHEMLoss(ignore_index=255, numel_frac=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    model = convert_syncbn_model(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    model = DDP(model)

    trainer = create_trainer(model, optimizer, loss_fn, device)
    ProgressBar(persist=False).attach(trainer, ['loss'])
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda _engine: scheduler.step())

    evaluator = create_evaluator(model, loss_fn, 19, device=device)

    if local_rank == 0:
        checkpointer = ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix=args.model,
            score_name='miou',
            score_function=lambda engine: engine.state.metrics['mIOU'],
            n_saved=5,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.COMPLETED, checkpointer,
            to_save={'model': model if not args.distributed else model.module})

    @trainer.on(Events.EPOCH_COMPLETED(every=2))
    def _evaluate(engine):
        state = evaluator.run(val_loader)
        if local_rank == 0:
            print("Epoch {}: {}"
                  .format(trainer.state.epoch, state.metrics))

    trainer.run(train_loader, max_epochs=args.epochs)
