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

da = aug.DataAugmentor()

# +
ims, seg_masks, bboxes = [],[],[]
for im_id in train_ids[:16]:
    im = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
    im = np.array(im, dtype=np.uint8)
    ims.append(im)
    
    bbs = utils.get_bbs(df_summary, im_id)
    bboxes.append(bbs)  
    
    seg_masks.append(utils.segmentation_heat_map(im, bbs))
    

fig, axs = plt.subplots(figsize=(20, 40), nrows=4, ncols=2)

for i in range(4):
    _ = axs[i][0].imshow(ims[i])
    
    for bb in bboxes[i]:
        utils.draw_bb(axs[i][0], bb)
    
    _ = axs[i][1].imshow(seg_masks[i], cmap='gray', vmin=0)
    
# -

ims_aug, masks_aug, bboxes_aug = da.augment_batch(ims, seg_masks, bboxes)

# +
fig, axs = plt.subplots(figsize=(20, 40), nrows=4, ncols=2)

for i in range(4):
    _ = axs[i][0].imshow(ims_aug[i])
    
    for bb in bboxes_aug[i]:
        utils.draw_bb(axs[i][0], bb)
    
    _ = axs[i][1].imshow(masks_aug[i], cmap='gray', vmin=0)
# -



# +
im_aug, mask_aug, bbs_aug = da.augment_image(im_np, mask, bbs)
fig, axs = plt.subplots(figsize=(20, 10), ncols=2)
_ = axs[0].imshow(im_aug)
_ = axs[1].imshow(mask_aug, cmap='gray', vmin=0)

for bb in bbs_aug:
    utils.draw_bb(axs[0], bb)
# -

np.random.choice(np.arange(4), 4, replace=False)

list(range(0,16,4))

# ## Resize

# +
im_resized = aug.resize_im(im_np, 0.25)
bbs_resized = aug.resize_bboxes(bbs, im_np.shape, im_resized.shape)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_resized)

for bb in bbs_resized:
    utils.draw_bb(ax, bb)
# -



# +

im_resized, bbs_resized = da.random_crop_resize(im_np, bbs)

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

# ## Color

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(aug.adjust_color(im_np, -0.25, 0.2, -.2, .3))


# ## Blur and sharpen

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(aug.blur(im_np, 0.02))

# ## Random Crop

# +
crop_dims = aug.random_crop_dims(im_np.shape, 700, 900)
im_cropped = aug.crop(im_np, *crop_dims)
bbs_cropped = aug.crop_box(bbs, *crop_dims)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_cropped)

for bb in bbs_cropped:
    utils.draw_bb(ax, bb)
# -



# +

fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.hist(np.random.rand(10000)**0.25)

# -

da = aug.DataAugmentor()

slices = da._random_puzzle_dims(im_np.shape)

aug.random_crop_dims(im_np.shape, 500, 300)

puzzle_dims = da._random_puzzle_dims(1024, 1024)

# +
puzzle_pieces = []
bbs_puzzle = []



for i, (im_id, (x, y, w, h)) in enumerate(zip(train_ids[:4], puzzle_dims)):
    im_pil = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
    im = np.array(im_pil, dtype=np.uint8)
    bbs = utils.get_bbs(df_summary, im_id)

    puzzle_pieces.append(aug.crop(im, x, y, w, h))
    bbs = aug.crop_box(bbs, x, y, w, h)
    if i%2 == 1:
        bbs[:, 0] = bbs[:, 0] + max(puzzle_dims[:, 0])
        
    if i >= 2:
        bbs[:, 1] += np.max(puzzle_dims[:, 1])
    
    bbs_puzzle.append(bbs)

im_top = np.hstack(puzzle_pieces[:2])
im_bottom = np.hstack(puzzle_pieces[2:])
im_puzzle = np.vstack((im_top, im_bottom))
bbs_puzzle = np.concatenate(bbs_puzzle)

fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.imshow(im_puzzle)

for bb in bbs_puzzle:
    utils.draw_bb(ax, bb)
# -

np.array(puzzle_dims)

for i in range(4):
    print (i%2)


