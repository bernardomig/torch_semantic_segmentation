import torch
import numpy as np
from time import time


def benchmark_model(model, batch, iterations, warmup):
    record = np.zeros(iterations)

    # throw some iterations for the initialization
    with torch.set_grad_enabled(False):
        for _ in range(warmup):
            _ = model(batch)

    # start performing the benchmark
    with torch.set_grad_enabled(False):
        for it in range(iterations):
            start = time()
            _ = model(batch)
            end = time()
            record[it] = end - start

    # calculate the statistics
    return {
        'fps': 1 / np.mean(record),
        'min': np.min(record),
        'max': np.max(record),
        'mean': np.mean(record),
        'std': np.std(record),
    }
