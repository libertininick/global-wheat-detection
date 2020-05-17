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
#     display_name: Python 3
#     language: python
#     name: python3
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

wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)

p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)
        
import global_wheat_detection.scripts.utils as utils
# -

# # Data

# +
DATA_PATH = 'C:/Users/liber/Dropbox/Python_Code/global_wheat_detection/data'

df_summary = pd.read_csv(f'{DATA_PATH}/train.csv')
image_ids = np.unique(df_summary['image_id'].values)



# +
im_id = image_ids[6]
im_pil = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
im_np = np.array(im_pil, dtype=np.uint8)
print(im_np.shape)

bbs = utils.get_bbs(df_summary, im_id)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_np)

for bb in bbs:
    utils.draw_bb(ax, bb)
# -

# # Segment objects

# +
mask = utils.bbs_to_segmentation_mask(im_np, bbs)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(mask, cmap='gray')

for bb in bbs:
    utils.draw_bb(ax, bb)
# -

# # Save masks

for i, im_id in enumerate(image_ids):

    im_pil = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
    im_np = np.array(im_pil, dtype=np.uint8)
    bbs = utils.get_bbs(df_summary, im_id)
    
    mask = utils.bbs_to_segmentation_mask(im_np, bbs)
    
    scipy.sparse.save_npz(f'../data/train_masks/{im_id}.npz', scipy.sparse.csc_matrix(mask))
    
    if (i+1)%100 == 0:
        print(i+1)
