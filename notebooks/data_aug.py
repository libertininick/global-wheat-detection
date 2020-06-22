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

# +
batch_size=8
resolution_out=256

(x, *y), (ims_aug, bboxes_aug) = loader.load_batch(batch_size=batch_size, resolution_out=resolution_out)
y_n_bboxes, y_segmentation, y_bboxes, bbox_class_wts = y
# -

print(x.shape)
print(y_n_bboxes.shape, torch.mean(y_n_bboxes), torch.std(y_n_bboxes))
print(y_segmentation.shape, torch.mean(y_segmentation), torch.std(y_segmentation))
print(torch.sum(y_segmentation > 0, dim=(1,2,3))/y_n_bboxes.squeeze())
b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
print(y_bboxes.shape, torch.mean(y_bboxes[b,1:,i,j]), torch.std(y_bboxes[b,1:,i,j]))
print(bbox_class_wts.shape, torch.sum(bbox_class_wts, dim=(1,2)))

# ### Visualize Augmentations

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)

for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_segmentation[i][0], cmap='gray', vmin=0)
# -

# ### Centroid mask

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)

for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_bboxes[i][0], cmap='gray', vmin=0, vmax=1)
# -

# ### Centroid classification wts

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)

for i in range(batch_size):
    _ = axs[i][0].imshow(y_bboxes[i][0], cmap='gray', vmin=0, vmax=1)
    _ = axs[i][1].imshow(bbox_class_wts[i]**0.1, cmap='gray')
# -

# ### `bbox_pred_to_dims`

for b in range(batch_size):
    bb_true = bboxes_aug[b][np.argsort(bboxes_aug[b][:,0])].astype(np.int64)
    
    # Predicted bbs
    i,j = np.where(y_bboxes[b,0,:,:] == 1)
    bboxes_pred = [utils.bbox_pred_to_dims(*bbox, resolution_out, resolution_out) 
                   for bbox 
                   in y_bboxes[b,1:,i, j].T
                  ]
    bboxes_pred = np.array(bboxes_pred)
    bboxes_pred = bboxes_pred[np.argsort(bboxes_pred[:,0])]
    
    print(b, 'Avg Percision:', utils.average_precision(bboxes_pred, bb_true))
    for idx in range(bboxes_pred.shape[0]):
        print(f'{idx:>3}', bb_true[idx], bboxes_pred[idx])
        
    print()


