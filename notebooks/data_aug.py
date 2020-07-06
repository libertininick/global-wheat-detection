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
import numpy as np
import pandas as pd
from PIL import Image
import scipy.sparse
from sklearn.cluster import KMeans
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
import global_wheat_detection.scripts.training_utils as training_utils
# -

# # Data Loader

# +
DATA_PATH = 'C:/Users/liber/Dropbox/Python_Code/global_wheat_detection/data'

loader = pp.DataLoader(path=DATA_PATH, seed=123)
# -

# ## Load tensors

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

# ### Visualize Augmentations

# +
fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)

for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_seg[i][0], cmap='gray', vmin=0)
# -

# ### Bounding box spread

fig, axs = plt.subplots(figsize=(20, 10*batch_size), nrows=batch_size, ncols=2)
for i in range(batch_size):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i][0], bb)
    
    _ = axs[i][1].imshow(y_bbox_spread[i][0], cmap='gray', vmin=0)

# + [markdown] heading_collapsed=true
# # Clustering

# + hidden=true
h, w = 256, 256
n_bboxes = 40
centroid_idxs = np.random.randint(low=0, high=h, size=(n_bboxes,2))

idxs = itertools.product(np.arange(h), np.arange(w))
idxs = np.array(list(idxs))

yh = scipy.spatial.distance_matrix(idxs, centroid_idxs)
yh = np.min(yh, axis=1)
yh = (np.argsort(np.argsort(-yh)))**0.5
yh = yh.reshape(h,w)/np.max(yh)
yh += (np.random.rand(h,w)-0.5)
yh = np.minimum(np.maximum(0, yh),1)

# + hidden=true
kernel = np.ones((5,5), np.uint8)
yh_smooth = cv2.morphologyEx((yh > 0.5).astype(np.uint8), cv2.MORPH_OPEN, kernel)
yh_smooth = cv2.morphologyEx(yh_smooth, cv2.MORPH_CLOSE, kernel)

i,j = np.where(yh_smooth == 1)
yh_pos = np.stack((i,j), axis=-1)

connectivity = 4
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yh_smooth, connectivity, cv2.CV_32S)

label_scale = (np.max(yh_pos) - np.min(yh_pos))/(np.max(labels) - np.min(labels))
cc = labels[i,j]*label_scale
yh_clustering = np.hstack((yh_pos, cc[:,None]))

# + hidden=true
kmeans = KMeans(n_clusters=n_bboxes, n_init=20, random_state=0).fit(yh_clustering[:,:2])
yh_centroids = np.round(kmeans.cluster_centers_[:,:2]).astype(np.int64)

# + hidden=true
fig, axs = plt.subplots(figsize=(15,7), ncols=2)
_ = axs[0].imshow(yh, cmap='gray', vmin=0, vmax=1)
_ = axs[1].imshow(yh_smooth, cmap='gray', vmin=0, vmax=1)

for y,x in centroid_idxs:
    rect = patches.Rectangle((x,y), 5, 5, edgecolor='black', facecolor='yellow')
    axs[0].add_patch(rect)
    rect = patches.Rectangle((x,y), 5, 5, edgecolor='black', facecolor='yellow')
    axs[1].add_patch(rect)
    
for y,x in yh_centroids:
    rect = patches.Rectangle((x,y), 5, 5, edgecolor='red', facecolor='red', alpha=0.5)
    axs[1].add_patch(rect)
# -

# # `bbox_pred_to_dims`

bboxes = training_utils.inference_output(y_n_bboxes, y_bbox_spread, y_seg, y_bboxes[:,1:,:,:], 224, 224)

fig, axs = plt.subplots(figsize=(10, 10*batch_size), nrows=batch_size)
for i in range(batch_size):
    _ = axs[i].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bboxes(axs[i], bb)
        
    for bb in bboxes[i][:,1:]:
        utils.draw_bboxes(axs[i], bb, color='yellow')


