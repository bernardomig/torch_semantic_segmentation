import os
import argparse

import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from apex import amp
from apex.parallel import (
    DistributedDataParallel, convert_syncbn_model,
)

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import create_lr_scheduler_with_warmup, CosineAnnealingScheduler

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from torch_semantic_segmentation.engine import (
    create_segmentation_trainer, create_segmentation_evaluator,
)
from torch_semantic_segmentation.data import (
    CityScapesDataset
)

from torch_semantic_segmentation.losses import OHEMLoss
from torch_semantic_segmentation.models.esnet import ESNet

DATASET_DIR = os.environ['DATASET_DIR']
if not os.path.isdir(DATASET_DIR):
    from sys import exit
    print("DATASET_DIR is invalid. Please specity one using environment variables.")
    exit(-1)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, required=True)

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
    albu.RandomCrop(768, 768),
    albu.HorizontalFlip(),
    albu.RandomBrightnessContrast(),
    albu.Normalize(),
    ToTensor(),
])
val_tfms = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])


cityscapes_dir = os.path.join(DATASET_DIR, 'cityscapes')
train_dataset = CityScapesDataset(
    cityscapes_dir, split='train', transforms=train_tfms)
val_dataset = CityScapesDataset(
    cityscapes_dir, split='val', transforms=val_tfms)


if args.distributed:
    kwargs = dict(num_replicas=world_size, rank=local_rank)
    train_sampler = DistributedSampler(train_dataset, **kwargs)
    kwargs['shuffle'] = False
    val_sampler = DistributedSampler(val_dataset, **kwargs)
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


model = ESNet(3, 19)
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
    args.learning_rate, args.learning_rate / 100,
    args.epochs * len(train_loader),
)
scheduler = create_lr_scheduler_with_warmup(
    scheduler, 0, args.learning_rate, 1000)


model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
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


if local_rank == 0:
    from time import localtime, strftime
    dirname = strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())
    dirname = 'checkpoints/esnet/{}'.format(dirname)

    checkpointer = ModelCheckpoint(
        dirname=dirname,
        filename_prefix='esnet',
        score_name='miou',
        score_function=lambda engine: engine.state.metrics['miou'],
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpointer,
        to_save={'model': model if not args.distributed else model.module},
    )


@trainer.on(Events.EPOCH_COMPLETED(every=2))
def _evaluate(_engine):
    state = evaluator.run(val_loader)
    if local_rank == 0:
        print("Epoch {}: {}"
              .format(trainer.state.epoch, state.metrics['miou']))


trainer.run(train_loader, max_epochs=args.epochs)
