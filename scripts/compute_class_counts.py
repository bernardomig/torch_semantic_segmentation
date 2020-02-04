import torch
from tqdm import tqdm
from torch_semantic_segmentation.data import CityScapesDataset
from torch.utils.data import DataLoader

ds = CityScapesDataset('/home/bml/datasets/cities-scapes',
                       split='train')
loader = DataLoader(ds, batch_size=20, num_workers=8)

counts = torch.zeros(256, dtype=torch.int64)


for _, label in tqdm(loader):
    counts += torch.bincount(label.flatten())

counts = counts.float() / len(loader)
counts = counts[:19]
counts = counts / counts.sum()

print(counts)

torch.save(counts, 'city-weights.pth')
