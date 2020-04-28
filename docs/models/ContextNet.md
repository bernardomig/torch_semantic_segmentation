# ContextNet

Based on the work __"ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time"__, by R. Poudel, U. Bonde, S. Liwicki and C. Zach ([arxiv](https://arxiv.org/abs/1805.04554)).

## Description

This model proposes a new neural network which is build on the concepts of the factorized convolution, network compression and pyramid representation, to produce semantic segmentation in real-time and with low memory requirements. To achieve this, the ContextNet is a two-branch network, with a deep network branch to capture global context information and a shallow branch that focuses on high-resolution segmentation details.

Keywords: Branch networks, network compression, pyramid networks.

### Flavours

| Name                | #Parameters | Frames/sec (1) |
| ------------------- | ----------- | -------------- |
| ContextNet18        | 0.850M      | 150.0          |
| ContextNet14        | 0.850M      | 135.5          |
| ContextNet12        | 0.850M      | 62.61          |
| ContextNet14-500(2) | 0.500M      |                |
| ContextNet14-160(2) | 0.160M      |                |

1. The performance benchmark for frames/sec was performed using a _NVIDIA RTX2080TI_, with a batch-size of 1 and a image size of 1920x1080. Check the benchmark for your system with the `benchmark_contextnet.py` script.
2. Both ContextNet14-500 and ContextNet14-160 are implementation with lower memory footprint.

### Use guide

```py
from torch_semantic_segmentation.models.contextnet import (
    contextnet12, contextnet14, contextnet18)

# example for ContextNet14 with 19 classes
model = contextnet14(in_channels=3, out_channels=19)
```

## Pretrained Models

| Model        | Dataset    | mIOU  | Weights | JIT Model |
| ------------ | ---------- | ----- | ------- | --------- |
| ContextNet12 | Cityscapes | 0.624 |         |           |
| ContextNet14 | Cityscapes |       |         |           |
| ContextNet18 | Cityscapes |       |         |           |
| ContextNet12 | BDD100K    |       |         |           |
| ContextNet14 | BDD100K    | 0.503 |         |           |
| ContextNet18 | BDD100K    |       |         |           |


## Training

ContextNet can be trained as a regular semantic segmentation network, or it can be trained using the method provided by the authors (more accurate, more complicated).

### Standard training

To train the network with the standard method, use the script `scripts/contextnet/train_contextnet.py` as follows (for the bdd100k dataset, not distributed):

```
python scripts/contextnet/train_contextnet.py \
    --batch_size 16             \
    --learning_rate 3e-3        \
    --epochs 1000               \
    --weight_decay 1e-5         \
    --crop_size 768x512         \
    --dataset bdd100k
```

### Training with deep compression and network prunning

Network prunning is used to reduce the network of network parameters and has effect on the generalization and overfitting of the model. Following [the lottery ticket hypothesys](https://arxiv.org/abs/1803.03635), we start with a larger network (twice the number of feature maps) and then decrease the number of parameters progressively by prunning to 1.5, 1.25 and 1 times the original size. The filters are prunned based on the lowest `l1` sum. This training has the effect of improoving the mIOU by about 2% in the Cityscapes dataset.

> TODO

## Evaluation

> TODO

## Benchmarking

To benchmark the model in a new hardware, run the script `scripts/contextnet/benchmark_contextnet.py` as follows:

```
python benchmark_contextnet.py \
    --size 2048x1024 \
    --device cuda:0 \
    --iterations 50 \
    --warmup 5      \
    --models contextnet12 contextnet14 contextnet18
```
