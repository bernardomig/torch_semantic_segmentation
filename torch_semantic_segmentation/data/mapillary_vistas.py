import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class MapillaryVistasDataset(Dataset):
    NUM_CLASSES = 66
    IGNORE_INDEX = 66

    def __init__(self, root_dir, split='train', transforms=None):
        split = ({
            'train': 'training',
            'valid': 'validation',
        })[split]

        images_dir = os.path.join(root_dir, split, 'images')
        labels_dir = os.path.join(root_dir, split, 'labels')

        ids = os.listdir(images_dir)
        ids = [os.path.splitext(id)[0] for id in ids]

        self.examples = [
            {
                'image': os.path.join(images_dir, f"{id}.jpg"),
                'label': os.path.join(labels_dir, f"{id}.png"),
            }
            for id in ids
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        label = example['label']

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        label[label > 65] = 65

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=label)
            return augmented['image'], augmented['mask'].long()
        else:
            return torch.from_numpy(image), torch.from_numpy(label).long()
