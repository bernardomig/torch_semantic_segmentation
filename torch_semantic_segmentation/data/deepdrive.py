import os
import cv2
import numpy as np
from numpy import array

from torch.utils.data import Dataset

CLASSES = array([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle',
])


class DeepDriveDataset(Dataset):

    def __init__(self, root_dir, split='train', transforms=None):
        images_dir = os.path.join(root_dir, 'images', split)
        labels_dir = os.path.join(root_dir, 'labels', split)

        self.examples = list(self._generate_examples(images_dir, labels_dir))

        self.transforms = transforms

    def _generate_examples(self, images_dir, labels_dir):
        images = os.listdir(images_dir)
        images = [os.path.join(images_dir, image) for image in images]
        images = sorted(images)
        labels = os.listdir(labels_dir)
        labels = [os.path.join(labels_dir, image) for image in labels]
        labels = sorted(labels)

        assert len(images) == len(labels)

        for image, label in zip(images, labels):
            yield {
                'image': image,
                'label': label,
            }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        label = example['label']

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

        augmented = self.transforms(image=image, mask=label)

        return augmented['image'], augmented['mask'].long()
