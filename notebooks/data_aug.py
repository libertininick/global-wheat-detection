# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python [conda env:wheat_env]
#     language: python
#     name: conda-env-wheat_env-py
# ---

# # Imports

# +
# %load_ext autoreload
# %autoreload 2

import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
import scipy.sparse
import torch
import torchvision

wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)

p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)
        
import global_wheat_detection.scripts.utils as utils
import global_wheat_detection.scripts.preprocessing as pp
# -

# # Data Loader

# +
DATA_PATH = 'C:/Users/liber/Dropbox/Python_Code/global_wheat_detection/data'

loader = pp.DataLoader(path=DATA_PATH, seed=123)
# -

# ## Visualize Augmentations

# +
ims_aug, masks_aug, bboxes_aug = loader._load_n_augment(batch_size=4, resolution_out=512, split='train')

fig, axs = plt.subplots(figsize=(20, 40), nrows=4, ncols=2)

for i in range(4):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(masks_aug[i], cmap='gray', vmin=0)
# -

# ## Load tensors

x, y_pretrained, y_segmentation, y_centroids, y_areas, y_sides = loader.load_batch(batch_size=4)

print(x.shape)
print(y_pretrained.shape)
print(y_segmentation.shape)
print(y_centroids.shape)
print(y_areas.shape)
print(y_sides.shape)
