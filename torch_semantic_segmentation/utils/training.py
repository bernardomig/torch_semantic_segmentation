import torch
from torch import distributed as dist


def setup_distributed(enable=True, local_rank=0):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)

    if enable:
        dist.init_process_group('nccl', init_method='env://')
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        local_rank = local_rank
    else:
        world_size = 1
        world_rank = 0
        local_rank = 0

    return world_size, world_rank, local_rank


def create_dataset(name):
    import os
    from functools import partial
    from logging import fatal

    from torch_semantic_segmentation.data import (
        CityScapesDataset,
        DeepDriveDataset,
    )

    dataset_dir = os.environ['DATASET_DIR']
    if not os.path.isdir(dataset_dir):
        fatal("DATASET_DIR is not specified or is invalid. Please specify it using the environment variable as `DATASET_DIR=/the/dataset/location/`.")

    def check_dataset_not_found(name, path):
        if not os.path.isdir(path):
            fatal(
                "{} dataset was not found. Please extract the dataset to {}."
                .format(name, path))

    if name == 'cityscapes':
        dataset_dir = os.path.join(dataset_dir, 'cityscapes')
        check_dataset_not_found('cityscapes', dataset_dir)
        return partial(CityScapesDataset, root_dir=dataset_dir)
    elif name == 'deepdrive':
        dataset_dir = os.path.join(dataset_dir, 'bdd100k/seg')
        check_dataset_not_found('deepdrive', dataset_dir)
        return partial(DeepDriveDataset, root_dir=dataset_dir)
    else:
        fatal("Unknown dataset {}.".format(name))


def create_sampler(dataset, world_size, local_rank, training=True, enable=True):
    from torch.utils.data import DistributedSampler

    if enable:
        return DistributedSampler(
            dataset,
            num_replicas=world_size, rank=local_rank,
            shuffle=training)

    else:
        return None
