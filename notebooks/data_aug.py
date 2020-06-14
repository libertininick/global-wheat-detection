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

x, y_pretrained, y_segmentation, y_bboxes, bbox_class_wts = loader.load_batch(batch_size=4, resolution_out=256)

print(x.shape)
print(y_pretrained.shape, torch.mean(y_pretrained), torch.std(y_pretrained))
print(y_segmentation.shape, torch.mean(y_segmentation), torch.std(y_segmentation))
b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
print(y_bboxes.shape, torch.mean(y_bboxes[b,1:,i,j]), torch.std(y_bboxes[b,1:,i,j]))
print(bbox_class_wts.shape, torch.sum(bbox_class_wts, dim=(1,2,3)))

# ### Centroid mask

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(y_bboxes[0,0,:,:], cmap='gray')

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(bbox_class_wts[0,0,:,:], cmap='gray')

i, j = np.where(y_bb_targets[0,0,:,:] == 1)
y_bb_targets[0,4, i, j]

# ### `bbox_pred_to_dims`

y_bb_targets[0,1:,i, j][:,0]

utils.bbox_pred_to_dims(*(y_bb_targets[0,1:,i, j][:,-1]), 256, 256)

utils.bbox_pred_to_dims(*[-4, 4, 4, 0], 1024, 1024)


