import os

from numpy import array
import cv2

from torch.utils.data import Dataset

CLASSES = array([
    'animal', 'archway', 'bicyclist', 'bridge',
    'building', 'car', 'cartluggagepram', 'child',
    'column_pole', 'fence', 'lanemkgsdriv', 'lanemkgsnondriv',
    'misc_text', 'motorcyclescooter', 'othermoving', 'parkingblock',
    'pedestrian', 'road', 'roadshoulder', 'sidewalk',
    'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone',
    'trafficlight', 'train', 'tree', 'truck_bus',
    'tunnel', 'vegetationmisc', 'void', 'wall', ])
TRAIN_CLASSES = array([
    'void', 'bicyclist', 'building', 'car',
    'column_pole', 'fence', 'pedestrian', 'road',
    'sidewalk', 'signsymbol', 'sky', 'vegetationmisc', ])
TRAIN_MAPPING = array([
    6,  0,  1,  0,  2,  3,  0,  6,  4,  5, 7,
    7,  9,  3,  3,  8,  6, 7,  7,  8, 9, 10,
    3,  0,  0,  3, 11,  3,  0, 11,  0,  2])


class CamvidDataset(Dataset):

    def __init__(self, root_dir, split='train', transforms=None):
        images_dir = os.path.join(root_dir, split, 'images')
        labels_dir = os.path.join(root_dir, split, 'labels')

        self.transforms = transforms

        self.examples = list(self._generate_examples(images_dir, labels_dir))

    def _generate_examples(self, images_dir, labels_dir):
        for f in os.listdir(images_dir):
            f, _ = os.path.splitext(f)

            yield {
                'image': os.path.join(images_dir, f'{f}.png'),
                'mask': os.path.join(labels_dir, f'{f}_P.png'),
            }

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        label = example['label']

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        label = TRAIN_MAPPING[label]

        augmented = self.transforms(image=image, mask=label)

        return augmented['image'], augmented['mask'].long()

    def __len__(self):
        return len(self.examples)
