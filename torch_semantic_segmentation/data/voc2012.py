import os

from numpy import array
import cv2

from torch.utils.data import Dataset


class VOC2012Dataset(Dataset):
    """The Pascal VOC 2012 Dataset
    """

    def __init__(self, root_dir, split='train', transforms=None):
        split_ids_file = os.path.join(
            root_dir, 'ImageSets', 'Segmentation', split + '.txt')
        split_ids = open(split_ids_file).readlines()
        split_ids = [id.strip() for id in split_ids]

        images_dir = os.path.join(root_dir, 'JPEGImages')
        labels_dir = os.path.join(root_dir, 'SegmentationClass')

        self.examples = [
            {'image': os.path.join(images_dir, id + '.jpg'),
             'mask': os.path.join(labels_dir, id + '.png')}
            for id in split_ids
        ]

        self.tfms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        mask = example['mask']

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tmfs(image=image, mask=mask)
            return augmented['image'], augmented['mask'].long()
        else:
            return image, mask
