import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage, ConfusionMatrix, Average, mIoU
from torch_semantic_segmentation.metrics import accuracy

from torch_semantic_segmentation.models import ENet
from torch_semantic_segmentation.data import CityScapesDataset

import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

from tqdm import tqdm

import argparse


def create_evaluator(model, loss_fn, device):
    def prepare_batch(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def validate_fn(trainer, batch):
        model.eval()
        x, y = prepare_batch(batch)
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y

    evaluator = Engine(validate_fn)
    Loss(loss_fn).attach(evaluator, 'loss')
    mIoU(ConfusionMatrix(num_classes=19)).attach(evaluator, 'mIOU')
    Average(output_transform=lambda x: accuracy(x[0], x[1], ignore_index=255)) \
        .attach(evaluator, 'accuracy')

    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str,
                        help='the state dict to load the weights from',
                        required=True)
    args = parser.parse_args()

    val_tfms = albu.Compose([
        albu.Resize(256, 512),
        albu.Normalize(),
        ToTensor(),
    ])

    val_ds = [
        CityScapesDataset(
            '/home/bml/datasets/cities-scapes',
            split='train', transforms=val_tfms),
        CityScapesDataset(
            '/home/bml/datasets/cities-scapes',
            split='val', transforms=val_tfms),
    ]

    val_loaders = [
        torch.utils.data.DataLoader(
            ds, batch_size=2, num_workers=8, drop_last=False)
        for ds in val_ds]

    device = torch.device('cuda')

    model = ENet(3, 19)
    model.load_state_dict(torch.load(args.state_dict, map_location='cpu'))
    model = model.to(device)

    counts = torch.load('city-weights.pth')[:19]
    weight = 1. / torch.log(1.02 + counts)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=255, weight=weight).to(device)

    evaluator = create_evaluator(model, loss_fn, device)

    state = evaluator.run(tqdm(val_loaders[0]))
    print('train metrics: ', state.metrics)
    state = evaluator.run(tqdm(val_loaders[1]))
    print('val metrics: ', state.metrics)
