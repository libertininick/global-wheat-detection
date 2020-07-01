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
# -

# # Data Loader

DATA_PATH = os.path.abspath('../data')
loader = DataLoader(path=DATA_PATH, seed=123)

# # Model testing

m = modules.WheatHeadDetector()

# ## Data

batch_size=24
resolution_out=128
x, y, (ims, bboxes) = loader.load_batch(batch_size=batch_size, resolution_out=resolution_out, split='train')

# ## Training

# +
yh = m._forward_train(x)
yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes = yh

with torch.no_grad():
    print(yh_n_bboxes.shape, torch.mean(yh_n_bboxes), torch.std(yh_n_bboxes))
    print(yh_bbox_spread.shape, torch.mean(yh_bbox_spread), torch.std(yh_bbox_spread))
    print(yh_seg.shape, torch.mean(yh_seg), torch.std(yh_seg))
    print(yh_bboxes.shape, torch.mean(yh_bboxes), torch.std(yh_bboxes))
# -



training_utils.training_loss(*yh, *y)

# ## Inference

# +
yh = m._forward_inference(x)

bboxes_pred = training_utils.inference_output(*yh, *list(x.shape[2:]))

for b in range(batch_size):
    print(len(bboxes[b]), len(bboxes_pred[b]))

utils.mAP(bboxes_pred, bboxes)
# -

# # Test training

model = modules.WheatHeadDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

for i in range(1200):
    print(i, end=' ')
    
    resolution_out, batch_size = random.choice([(512, 2), (400, 2), (300, 4), (256, 4)])
    
    x, y, (ims, bboxes) = loader.load_batch(batch_size=batch_size, resolution_out=resolution_out)
    yh = model._forward_train(x)

    loss = training_utils.training_loss(*yh, *y)
    #print(f'{loss.item():.2f}')
    
    loss.backward()
    
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr*batch_size # adj LR for batch size
    optimizer.step()
    optimizer.param_groups[0]['lr'] = lr
    
    optimizer.zero_grad()

# # Testing

x, y, (ims, bboxes) = loader.load_batch(batch_size=4, resolution_out=512)

yh = model._forward_inference(x)
bboxes_pred = training_utils.inference_output(*yh, *list(x.shape[2:]))

yh[2][0,0]

# +
fig, axs = plt.subplots(figsize=(20, 40), nrows=4, ncols=2)

for i in range(4):
    _ = axs[i][0].imshow(ims[i])
    
    for bb in bboxes[i]:
        utils.draw_bboxes(axs[i][0], bb)
        
    _ = axs[i][1].imshow(torch.sigmoid(yh[1][i][0])*torch.sigmoid(yh[2][i,0]), vmin=0, vmax=1)
# -



utils.mAP(bboxes_pred, bboxes_aug)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(yh[0, 0, :, :])

bboxes_pred

# +
fig, ax = plt.subplots(figsize=(10,10))

_ = ax.imshow()
# -


