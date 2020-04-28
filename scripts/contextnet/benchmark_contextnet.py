import argparse
import torch
from tabulate import tabulate

from torch_semantic_segmentation.utils.benchmark import benchmark_model
from torch_semantic_segmentation.models.contextnet import (
    contextnet12, contextnet14, contextnet18)


def image_size(size):
    size = size.split('x')
    if len(size) == 1:
        width = height = int(size[0])
        return width, height
    elif len(size) == 2:
        return int(size[0]), int(size[1])

    else:
        raise ValueError


parser = argparse.ArgumentParser()
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--iterations', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--models',
                    nargs='+',
                    default=['contextnet12', 'contextnet14', 'contextnet18'])
parser.add_argument('--num_classes', default=19, type=int)
parser.add_argument('--size', default='1920x1080', type=image_size)
parser.add_argument('--dtype', default='float32',
                    choices=['float16', 'float32'])
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--format', required=False)
args = parser.parse_args()

device = torch.device(args.device)
dtype = (torch.float32 if args.dtype == 'float32' else torch.float16)

batch_size = args.batch_size
width, height = args.size
batch = torch.randn((batch_size, 3, width, height), device=device, dtype=dtype)


def create_model(name, in_channels, out_channels):
    if name == 'contextnet12':
        return contextnet12(in_channels, out_channels)
    elif name == 'contextnet14':
        return contextnet14(in_channels, out_channels)
    elif name == 'contextnet18':
        return contextnet18(in_channels, out_channels)
    else:
        raise ValueError("Model not known.")


results = []

for name in args.models:
    num_classes = args.num_classes

    model = create_model(name, 3, num_classes)
    model = model.to(device).to(dtype)

    result = benchmark_model(model, batch, args.iterations, args.warmup)
    result = [name, *[result[metric]
                      for metric in ['fps', 'mean', 'std', 'min', 'max']]]
    results.append(result)

results = tabulate(results,
                   floatfmt='.3f',
                   headers=[
                       'model', 'fps (mean)', 'time (mean)', 'time (std)', 'time (min)', 'time (max)'],
                   tablefmt=args.format)

print(results)
