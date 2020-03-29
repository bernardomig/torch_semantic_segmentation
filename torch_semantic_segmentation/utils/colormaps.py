"""Includes some of the colormaps for visualization of the masks
and predictions using matplotlib.
"""

import numpy as np
from matplotlib.colors import ListedColormap


def create_pascal_colormap():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return ListedColormap(colormap / 255., N=512)


def create_cityscapes_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:19, :] = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])
    return ListedColormap(colormap / 255., N=256)


def create_camvid_colormap():
    colormap = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [0.3, 0.3, 0.3],
        [255, 255, 255],
        [0, 0, 0],
        [30, 144,  255],
        [139, 69,   19],
        [202, 255, 112],
        [255, 20,  147],
        [128, 0,   128],
        [240, 230, 140],
        [255, 215,   0]])
    return ListedColormap(colormap / 255., N=12)
