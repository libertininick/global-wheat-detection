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

import itertools
import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from PIL import Image
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

# # Load tensors

# +
batch_size = 4

x, y, (ims_aug, bboxes_aug) = loader.load_batch(batch_size=batch_size, split='train')

if y is not None:
    y_n_bboxes, y_bbox_spread, y_seg, y_bboxes = y

# +
print(x.shape)

print(y_n_bboxes.shape, torch.mean(y_n_bboxes), torch.std(y_n_bboxes))
print(y_bbox_spread.shape, torch.sum(y_bbox_spread, dim=(-2,-1)).squeeze())

print(y_seg.shape, torch.mean(y_seg))

b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
print(y_bboxes.shape, torch.mean(y_bboxes[b,1:,i,j]), torch.std(y_bboxes[b,1:,i,j]))


# -

# ## Visualize Augmentations

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)

for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_seg[i][0], cmap='gray', vmin=0)
    
_ = axs[0][0].set_title(label='Ground Truth', loc='left', fontdict={'fontsize': 16})
_ = axs[0][1].set_title(label='Segmentation', loc='left', fontdict={'fontsize': 16})
# -

# ## Bounding box spread

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)
for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_bbox_spread[i][0], cmap='gray', vmin=0)
    
    # Add probability labels to the distribution grid
    for x in range(8):
        for y in range(8):
            axs[i][1].text(x,y,f'{y_bbox_spread[i][0][y,x]:.2%}',color='r',ha='center',va='center')
    
_ = axs[0][0].set_title(label='Ground Truth', loc='left', fontdict={'fontsize': 16})
_ = axs[0][1].set_title(label='BBox Distribution', loc='left', fontdict={'fontsize': 16})
