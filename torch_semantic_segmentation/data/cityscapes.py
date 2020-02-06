import os
import cv2
import numpy as np
import torch
from numpy import array

from torch.utils.data import Dataset

CITIES = {
    'train': ["jena", "zurich", "weimar", "ulm", "tubingen", "stuttgart",
              "strasbourg", "monchengladbach", "krefeld", "hanover",
              "hamburg", "erfurt", "dusseldorf", "darmstadt", "cologne",
              "bremen", "bochum", "aachen"],
    'val': ["frankfurt", "munster", "lindau"], }
CLASSES = array([
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi',
    'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking',
    'rail track', 'building', 'wall', 'fence', 'guard rail',
    'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train',
    'motorcycle', 'bicycle', 'license plate'])
TRAIN_MAPPING = array([
    255, 255, 255, 255, 255, 255, 255,   0,  1, 255, 255,   2,  3,  4, 255,
    255, 255,   5, 255,   6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 255,
    255,  16, 17, 18, 255, ])
CLASS_FREQ = array([
    0.3687, 0.0608, 0.2282, 0.0066, 0.0088, 0.0123, 0.0021, 0.0055, 0.1593,
    0.0116, 0.0402, 0.0122, 0.0014, 0.0699, 0.0027, 0.0024, 0.0023, 0.0010,
    0.0041,
])


class CityScapesDataset(Dataset):

    def __init__(self, root_dir, split='train', transforms=None):
        cities = CITIES[split]
        images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        labels_dir = os.path.join(root_dir, 'gtFine', split)

        self.transforms = transforms

        self.examples = list(self._generate_examples(
            cities, images_dir, labels_dir))

    def _generate_examples(self, cities, images_dir, labels_dir):
        for city in cities:
            city_dir = os.path.join(images_dir, city)

            for f in os.listdir(city_dir):
                id = f[:-16]
                img = os.path.join(images_dir, city, f)
                label = os.path.join(
                    labels_dir, city, id + '_gtFine_labelIds.png')
                yield {
                    'image': img,
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
        label = TRAIN_MAPPING[label]

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=label)
            return augmented['image'], augmented['mask'].long()
        else:
            return torch.from_numpy(image), torch.from_numpy(label).long()
