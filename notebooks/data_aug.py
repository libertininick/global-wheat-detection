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
import global_wheat_detection.scripts.data_augmentation as aug
# -

# # Data

# +
DATA_PATH = 'C:/Users/liber/Dropbox/Python_Code/global_wheat_detection/data'

df_summary = pd.read_csv(f'{DATA_PATH}/train.csv')

image_ids = dict()
for _, (im_id, *o, source) in df_summary.iterrows():
    image_ids[im_id] = source  

train_ids = list(image_ids.keys())[:-373]
holdout_ids = list(image_ids.keys())[-373:]
# -

# # Data Augmentation

# +
im_id = train_ids[0]
im_pil = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
im_np = np.array(im_pil, dtype=np.uint8)

bbs = utils.get_bbs(df_summary, im_id)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_np)

for bb in bbs:
    utils.draw_bb(ax, bb)
# -

# ## Resize

# +
im_resized = aug.resize_im(im_np, 0.25)
bbs_resized = aug.resize_bboxes(bbs, im_np.shape, im_resized.shape)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_resized)

for bb in bbs_resized:
    utils.draw_bb(ax, bb)
# -

# ## Rotate

# +
angle = 90
im_cropped, bbs_cropped = aug.rotate(im_np, bbs, angle)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_cropped)

for bb in bbs_cropped:
    utils.draw_bb(ax, bb)
# -




