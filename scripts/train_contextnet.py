import argparse
import os
from logging import info

import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model)

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import (
    create_lr_scheduler_with_warmup, CosineAnnealingScheduler)

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from torch_semantic_segmentation.models.contextnet import contextnet14
from torch_semantic_segmentation.engine import (
    create_segmentation_trainer, create_segmentation_evaluator)
from torch_semantic_segmentation.data import (
    CityScapesDataset, DeepDriveDataset, MapillaryVistasDataset)
from torch_semantic_segmentation.losses import OHEMLoss

from torch_semantic_segmentation.utils.training import *
from torch_semantic_segmentation.utils.logging import *

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--crop_size', type=int, default=768)
parser.add_argument('--state_dict', type=str, required=False)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

distributed = args.distributed
world_size, world_rank, local_rank = setup_distributed(
    distributed, args.local_rank)

if local_rank == 0:
    wandb.init(
        project='torch-semantic-segmentation',
        config=args,
        group='contextnet')

device = torch.device('cuda')

train_tfms = albu.Compose([
    albu.RandomScale([0.5, 2.0]),
    albu.RandomCrop(args.crop_size, args.crop_size),
    albu.HorizontalFlip(),
    albu.HueSaturationValue(),
    albu.Normalize(),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])

Dataset = create_dataset('cityscapes')
train_dataset = Dataset(split='train', transforms=train_tfms)
val_dataset = Dataset(split='val', transforms=val_tfms)

sampler_args = dict(world_size=world_size,
                    local_rank=local_rank,
                    enable=distributed)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=8,
    sampler=create_sampler(train_dataset, **sampler_args),
    shuffle=not distributed,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    sampler=create_sampler(val_dataset, training=False, **sampler_args),
)


model = contextnet14(3, 19)

if args.state_dict is not None:
    state_dict = torch.load(args.state_dict, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)


model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)

counts = torch.from_numpy(CityScapesDataset.CLASS_FREQ.astype('f4'))
weight = 1. / torch.log(1.02 + counts)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
loss_fn = loss_fn.cuda()


scheduler = CosineAnnealingScheduler(
    optimizer, 'lr',
    args.learning_rate, args.learning_rate / 1000,
    args.epochs * len(train_loader),
)
scheduler = create_lr_scheduler_with_warmup(
    scheduler, 0, args.learning_rate, 1000)


model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
if args.distributed:
    model = convert_syncbn_model(model)
    model = DistributedDataParallel(model)

trainer = create_segmentation_trainer(
    model, optimizer, loss_fn,
    device=device,
    use_f16=True,
    logging=local_rank == 0,
)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

evaluator = create_segmentation_evaluator(
    model,
    device=device,
    num_classes=19,
)


@trainer.on(Events.ITERATION_COMPLETED(every=400))
def evaluate(engine):
    evaluator.run(val_loader)


if local_rank == 0:

    checkpointer = ModelCheckpoint(
        dirname=os.path.join(wandb.run.dir, 'weights'),
        filename_prefix='contextnet',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )

    setup_logging(wandb.run.dir, trainer, evaluator, freq=100)

trainer.run(train_loader, max_epochs=args.epochs)
