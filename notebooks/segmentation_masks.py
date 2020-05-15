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
image_ids = df_summary['image_id'].values

holdout_ids = set(image_ids[-100:])
train_ids = sorted(list(set(image_ids).difference(holdout_ids)))


# +
im_id = train_ids[6]
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

# +
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow((im_np * (1 - mask[:,:,None])).astype(np.uint8))


# -

# # Load pre-trained model

resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
resnet50.eval()

modules=list(resnet50.children())
feature_model = nn.Sequential(*modules)[:6]

# # Summarize activations

# +
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_activations(model, im):
    input_tensor = preprocess(im)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch).squeeze(0).numpy()
        
    return output

def activation_ranges(x):
    lower = np.quantile(x, q=0.25, axis=(1,2))
    upper = np.quantile(x, q=0.75, axis=(1,2))
    
    return upper - lower


# +
avg_ranges = np.zeros(128)

for im_id in image_ids[:100]:
    im = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
    x = get_activations(feature_model, im)
    avg_ranges += activation_ranges(x)
    
avg_ranges /= 100
avg_ranges = (avg_ranges - np.mean(avg_ranges))/np.std(avg_ranges)
avg_ranges = 1/(1 + np.exp(-avg_ranges))

# +
avg_bb_activations = np.zeros(128) 
for im_id in image_ids[:100]:
    im = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
    x = get_activations(feature_model, im)

    bbs = get_bbs(df_summary, im_id)
    im_bg = Image.fromarray(np.uint8(color_mask_bbs(im, bbs)))
    
    xb = get_activations(feature_model, im_bg)
    
    avg_bb_activations += np.mean(np.abs(x - xb), axis=(1,2))
    
avg_bb_activations /= 100
avg_bb_activations = (avg_bb_activations - np.mean(avg_bb_activations))/np.std(avg_bb_activations)
avg_bb_activations = 1/(1 + np.exp(-avg_bb_activations))
# -

np.sort(avg_ranges*avg_bb_activations)

fig, ax = plt.subplots(figsize=(8, 8))
ax.bar(np.arange(len(avg_ranges)), np.sort(avg_ranges*avg_bb_activations), width=1)

# +
fig, axs = plt.subplots(figsize=(8, 8), nrows=2)

axs[0].bar(np.arange(len(avg_ranges)), avg_ranges, width=1)
axs[1].bar(np.arange(len(avg_ranges)), avg_bb_activations, width=1)


# -

def edge_pad(x):
    
    *o, h, w = list(x.shape)
    h_pad = h%2 == 1
    if h_pad:
        x = torch.cat((x, torch.zeros(*o, 1, w)), dim=2)
        
    *o, h, w = list(x.shape)
    w_pad = w%2 == 1
    if w_pad:
        x = torch.cat((x, torch.zeros(*o, h, 1)), dim=3)
        
    return x, (h_pad, w_pad)


in_channels=3
x = torch.randn(1, in_channels,100,150)
xp, (h_pad, w_pad) = edge_pad(x)

c_d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=2, stride=2, padding=0)
c_u = nn.ConvTranspose2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=2, stride=2, padding=0)

# +
with torch.no_grad():
    y = c_d(xp)
    x2 = c_u(y)
    
    if h_pad:
        x2 = x2[:, :, :-1, :]
    
    if w_pad:
        x2 = x2[:, :, :, :-1]

    
print(x.shape)
print(y.shape)
print(x2.shape)
# -

torch.zeros(1,1,1,7)

torch.cat((x, torch.zeros(1,in_channels,1,7)), dim=2).shape


def trim_n_segment(x, max_h, max_w):
    h, w = x.shape
    
    # Trim width
    xt = x[:, :max_w]
    
    # Segment height
    if h > max_h:
        div = int(np.ceil(h/max_h))
        h_seg = h//div
        seg_step = h_seg//2

        st_idxs = list(range(0, h, seg_step))
        st_idxs = st_idxs[:len(st_idxs)//2*2][:-1]
        max_len = max(h, st_idxs[-1] + h_seg) - st_idxs[-1]

        segment_idxs = [(st_idx, st_idx + max_len) for st_idx in st_idxs]
        segments = [xt[st_idx:end_idx, :] for st_idx, end_idx in segment_idxs]
    else:
        segment_idxs = [(0, h)]
        segments = [xt]
        
    return segment_idxs, segments


segment_idxs, segments = trim_n_segment(np.random.rand(123, 210), 100, 150)

# +
h = 200
max_h =100

div = int(np.ceil(h/max_h))
h_seg = h//div
seg_step = h_seg//2

st_idxs = list(range(0, h, seg_step))
st_idxs = st_idxs[:len(st_idxs)//2*2][:-1]
max_len = max(h, st_idxs[-1] + h_seg) - st_idxs[-1]

segment_idxs = [(st_idx, st_idx + max_len) for st_idx in st_idxs]

# -

segments[2].shape

h = 101


