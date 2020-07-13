# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:wheat_env] *
#     language: python
#     name: conda-env-wheat_env-py
# ---

# # Imports

# +
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
import tcod
import torch
import torch.nn as nn
from torchvision import transforms


wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)

p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)

from global_wheat_detection.scripts.preprocessing import DataLoader
import global_wheat_detection.scripts.modules as modules
import global_wheat_detection.scripts.utils as utils
import global_wheat_detection.scripts.training_utils as training_utils

DATA_PATH = os.path.abspath('../data')
MODEL_PATH = os.path.abspath('../models')
# -

# # Data Loader

loader = DataLoader(path=DATA_PATH, seed=123)

# # Model

model = modules.WheatHeadDetector()
model.load_state_dict(torch.load(f'{MODEL_PATH}/wheat_head_5.pth'))

batch_size=4
x, y, (ims, bboxes) = loader.load_batch(batch_size=batch_size, split='train')
y_n_bboxes, y_bbox_spread, y_seg, y_bboxes = y

# +
yh = model._forward_inference(x)
yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes = yh

n_bboxes = (yh_n_bboxes.numpy().squeeze()*6.07 + 16.65)**(1.33)

b, c, h, w = list(yh_bbox_spread.shape)
yh_bbox_spread = nn.Softmax(dim=-1)(yh_bbox_spread.view(b,c,-1)).view(b,c,h,w)

# +
im_idx = 0
print(n_bboxes[im_idx])

grid_ct = np.round(yh_bbox_spread[im_idx][0].numpy()*n_bboxes[im_idx],2)

p1 = torch.sigmoid(yh_seg[im_idx][0])
p2 = torch.sigmoid(yh_bboxes[im_idx][0])
p3 = p1*p2

fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(p3, cmap='gray', vmin=0, vmax=1)

step_size = 112//8

loc = plticker.MultipleLocator(base=step_size)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-')

# Add some labels to the gridsquares
for j in range(8):
    y=step_size/2 + j*step_size
    for i in range(8):
        x=step_size/2.+float(i)*step_size
        ax.text(x,y,f'{grid_ct[j,i]:.2f}',color='r',ha='center',va='center')
# -

# # Clustering

# ## Find "land"

land_threshold = 0.5**2

# +
land_is, land_js = np.where(p3 >= land_threshold)
land_mask = (p3.numpy() >= land_threshold).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(land_mask, 8, cv2.CV_32S)

island_centroids = np.round(centroids[1:,], 2).astype(np.int64)
island_centroids = island_centroids[:,::-1] # map x and y to i and j

# +
fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(land_mask, cmap='gray', vmin=0, vmax=1)

step_size = 112//8

loc = plticker.MultipleLocator(base=step_size)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-')

# Add some labels to the gridsquares
for j in range(8):
    y=step_size/2 + j*step_size
    for i in range(8):
        x=step_size/2.+float(i)*step_size
        ax.text(x,y,f'{grid_ct[j,i]:.2f}',color='r',ha='center',va='center')


# -

# ## Travel cost

def calc_cost(cost, path):
    i, j = path.T
    path_costs = cost[i,j][1:]
    
    return np.sum(path_costs) + len(path_costs)


cost = np.ceil(-np.log(p3.numpy())*20).astype(np.int64)
fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(cost, cmap='gray')

# ## Path finder

graph = tcod.path.SimpleGraph(cost=cost, cardinal=1, diagonal=1)

# +
pf = tcod.path.Pathfinder(graph)
pf.add_root((69,78))
i,j = pf.path_from((90,63)).T

path_mask = np.zeros_like(cost)
path_mask[i, j] = 1
fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(path_mask, cmap='gray')
# -

# ## Cost Matrix

# +
costs = []
for root_i, root_j in island_centroids:

    pf = tcod.path.Pathfinder(graph)
    pf.add_root((root_i, root_j))
    
    costs_to_root = []
    for i,j in zip(land_is, land_js):
        if (i,j) != (root_i, root_j):
            c = calc_cost(cost, pf.path_from((i,j)))
        else:
            c = 0
        costs_to_root.append(c)

    costs.append(costs_to_root)
    
costs = np.array(costs).T
# -

# ## Single linkage clustering

# +
cluster_candidates = defaultdict(list)

step_size = 112//8
steps = list(range(-step_size,112,step_size))

coverage_ct = np.zeros((8,8))
for i, step_i in enumerate(steps,-1):
    for j, step_j in enumerate(steps,-1):
        for stride in [1, 2]:
            k = np.round(np.sum(grid_ct[max(0,i):i+stride,max(0,j):j+stride]),0).astype(np.int)
            coverage_ct[max(0,i):i+stride,max(0,j):j+stride] += 1

            if k > 0:
                idx_mask = np.logical_and(
                      np.logical_and(land_is >= step_i, land_is < step_i+step_size*stride)
                    , np.logical_and(land_js >= step_j, land_js < step_j+step_size*stride)
                )
                mask_is = land_is[idx_mask]
                mask_js = land_js[idx_mask]

                d = costs[idx_mask, :]

                if d.shape[0] > 1:
                    z = linkage(d)
                    clusters = fcluster(z, t=k, criterion='maxclust')

                    for c in range(1,k+1):
                        c_is = mask_is[clusters==c]
                        c_js = mask_js[clusters==c]
                        c_probs = p3[c_is,c_js].numpy()
                        marker = np.argmax(c_probs)
                        cluster_candidates[c_is[marker],c_js[marker]].append(np.mean(c_probs).item())

# Full cluster
k = np.round(np.sum(grid_ct),0).astype(np.int)
z = linkage(costs)
clusters = fcluster(z, t=k, criterion='maxclust')
for c in range(1,k+1):
    c_is = land_is[clusters==c]
    c_js = land_js[clusters==c]
    c_probs = p3[c_is,c_js].numpy()
    marker = np.argmax(c_probs)
    cluster_candidates[c_is[marker],c_js[marker]].append(np.mean(c_probs).item())
coverage_ct += 1
    
cluster_candidates = [(k, (np.sum(v)/np.mean(coverage_ct))**0.5) for k, v in cluster_candidates.items()]
cluster_candidates.sort(key=lambda x: -x[1])
len(cluster_candidates)

# +
out_mask = np.zeros(land_mask.shape)
for (i,j), c in cluster_candidates[:20]:
    out_mask[i,j] = c

fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(out_mask, cmap='gray', vmin=0, vmax=1)

step_size = 112//8

loc = plticker.MultipleLocator(base=step_size)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-')

# Add some labels to the gridsquares
for j in range(8):
    y=step_size/2 + j*step_size
    for i in range(8):
        x=step_size/2.+float(i)*step_size
        ax.text(x,y,f'{grid_ct[j,i]:.2f}',color='r',ha='center',va='center')
