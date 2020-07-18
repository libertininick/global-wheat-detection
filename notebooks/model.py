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
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from PIL import Image
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

# # Model testing

m = modules.WheatHeadDetector()

# ## Data

batch_size=4
x, y, (ims, bboxes) = loader.load_batch(batch_size=batch_size, split='train')
y_n_bboxes, y_bbox_spread, y_seg, y_bboxes = y

# ## Training

# +
yh = m._forward_train(x)
yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes = yh

with torch.no_grad():
    print(yh_n_bboxes.shape, torch.mean(yh_n_bboxes), torch.std(yh_n_bboxes))
    print(yh_bbox_spread.shape, torch.mean(yh_bbox_spread), torch.std(yh_bbox_spread))
    print(yh_seg.shape, torch.mean(yh_seg), torch.std(yh_seg))
    print(yh_bboxes.shape, torch.mean(yh_bboxes), torch.std(yh_bboxes))
# +
(loss_n_bboxes
 , loss_bbox_spread
 , loss_segmentation
 , loss_bb_centroids
 , loss_bb_regressors
 , denom
) = training_utils.training_loss(*yh, *y)

print(loss_n_bboxes, loss_bbox_spread, loss_segmentation, loss_bb_centroids, loss_bb_regressors, denom)
# -

# ## Inference

yh = m._forward_inference(x)
yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes = yh

# # Training

model = modules.WheatHeadDetector()

lower_lr, upper_lr = 1e-5, 3e-3
lr_spread = (upper_lr - lower_lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lower_lr)

# +
losses = []
cycle_len = 120
n_cycles = 20
n = cycle_len*n_cycles
idxs = np.arange(n)
lr_scales = (np.sin(idxs/cycle_len*2*np.pi - 1.5) + 1)/2
batch_size = 4

loss_scales = np.ones(5)
for i in range(n):
    x, y, (ims, bboxes) = loader.load_batch(batch_size=batch_size)
    yh = model._forward_train(x)

    ( loss_n_bboxes
    , loss_bbox_spread
    , loss_segmentation
    , loss_bb_centroids
    , loss_bb_regressors
    , denom
    ) = training_utils.training_loss(*yh, *y)
    
    loss = (loss_n_bboxes*lr_scales[0]
            + loss_bbox_spread*lr_scales[1] 
            + loss_segmentation*lr_scales[2] 
            + loss_bb_centroids*lr_scales[3] 
            + loss_bb_regressors*lr_scales[4]
           )/denom
    losses.append((loss_n_bboxes.item()
                   , loss_bbox_spread.item()
                   , loss_segmentation.item()
                   , loss_bb_centroids.item()
                   , loss_bb_regressors.item()
                  ))
    
    loss.backward()
    
    optimizer.param_groups[0]['lr'] = lr_scales[i]*lr_spread + lower_lr
    optimizer.step()
    optimizer.zero_grad()
    
    if (i + 1)%(cycle_len) == 0:
        print(f'--{(i + 1)/n:.0%}', np.round(np.median(np.array(losses[-cycle_len:]), axis=0), 2))
        loss_scales = np.median(np.array(losses[-cycle_len:]), axis=0)
        loss_scales = (np.max(loss_scales)/loss_scales)**0.5
    else:
        if (i + 1)%(cycle_len//10) == 0:
            print(f'{(i + 1)%cycle_len/cycle_len:.0%}', end=' ')
# -

torch.save(model.state_dict(), f'{MODEL_PATH}/wheat_head_5.pth')
