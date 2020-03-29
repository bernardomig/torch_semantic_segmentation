import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
import torch.distributed as dist
import argparse
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

from torch_semantic_segmentation.models import ENet, FastSCNN
from torch_semantic_segmentation.data import CityScapesDataset, DeepDriveDataset
from torch_semantic_segmentation.losses import OHEMLoss

from apex import amp

logging.basicConfig(level=logging.INFO)

DATASET_DIR = '/srv/datasets/bdd100k/bdd100k/seg'


def create_tfms(crop_size, scaling, hard=False):
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2 as ToTensor
    hard_tfms = [
        albu.Rotate(limit=5),
        albu.GaussNoise(var_limit=(10, 50), p=0.2),
        albu.OneOf([
            albu.RandomBrightnessContrast(),
            albu.HueSaturationValue(),
        ])
    ]

    train_tfms = albu.Compose([
        albu.RandomScale(scaling),
        albu.RandomCrop(crop_size[1], crop_size[0]),
        albu.HorizontalFlip(),
        *hard_tfms,
        albu.Normalize(),
        ToTensor(),
    ])
    val_tfms = albu.Compose([
        albu.CenterCrop(704, 1280),
        albu.Normalize(),
        ToTensor(),
    ])
    return train_tfms, val_tfms


def create_trainer(model, optimizer, loss_fn, device, use_f16=False, logging=True):
    def prepare_batch(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def update_fn(_trainer, batch):
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


def train(gpu, args):
    rank = args.rank * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank,
    )

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    from torch import nn
    from torch_semantic_segmentation.wrappers import DeepSupervisionWrapper
    from torch_semantic_segmentation.models.fastscnn import Classifier

    model = FastSCNN(3, 19)
    model = DeepSupervisionWrapper(model, [
        (model.downsample, nn.Sequential(
            Classifier(64, 19),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )),
        (model.features, nn.Sequential(
            Classifier(128, 19),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),
        ))
    ])

    state_dict = torch.load(
        'weights/fastscnn_bdd100k_mIOU=0.461.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()

    batch_size = 16

    train_tfms, val_tfms = create_tfms(
        crop_size=[640, 480],
        scaling=[0.5, 2.])

    train_dataset = DeepDriveDataset(
        DATASET_DIR, split='train', transforms=train_tfms)
    val_dataset = DeepDriveDataset(
        DATASET_DIR, split='val', transforms=val_tfms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
    )

    ohem_fn = OHEMLoss(ignore_index=255, numel_frac=0.05)
    # counts = torch.from_numpy(CityScapesDataset.CLASS_FREQ.astype('f4'))
    # weight = 1. / torch.log(1.02 + counts)
    # ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
    ohem_fn = ohem_fn.cuda()

    def loss_fn(inputs, target):
        input, (aux1, aux2) = inputs
        return ohem_fn(input, target) + 0.4 * ohem_fn(aux1, target) + 0.4 * ohem_fn(aux2, target)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.85, weight_decay=1e-5, nesterov=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs * len(train_loader)
    # )

    model = convert_syncbn_model(model)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    model = DDP(model)

    trainer = create_trainer(model, optimizer, loss_fn,
                             device=torch.cuda.current_device(),
                             use_f16=True,
                             logging=rank == 0)

    @trainer.on(Events.ITERATION_COMPLETED)
    def step_scheduler(_trainer):
        scheduler.step()

    evaluator = create_evaluator(
        model, ohem_fn,
        num_classes=19,
        device=torch.cuda.current_device())

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate(engine):
        def format_metrics(metrics):
            return ', ' \
                .join([f"{name}={value}"
                       for name, value in state.metrics.items()
                       if name in ['loss', 'mIOU', 'accuracy']])

        state = evaluator.run(val_loader)
        if rank == 0:
            info("evaluation on val done: {}"
                 .format(format_metrics(state.metrics)))
            info("finished evaluation")

    if rank == 0:
        checkpoints_dir = 'checkpoints_distributed_fastscnn'
        info("logging checkpoints to {}".format(checkpoints_dir))
        checkpointer = Checkpoint(
            to_save={
                'model': model.module,
                'optimizer': optimizer,
                'amp': amp,
            },
            save_handler=DiskSaver(checkpoints_dir, create_dir=True),
            n_saved=5,
            filename_prefix='fastscnn',
            score_function=lambda _: evaluator.state.metrics['mIOU'],
            score_name='mIOU',
            global_step_transform=global_step_from_engine(trainer))
        evaluator.add_event_handler(Events.COMPLETED, checkpointer)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-r', '--rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '88881'
    mp.spawn(train, nprocs=args.gpus, args=(args,))
