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

import global_wheat_detection.scripts.utils as utils
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



# +
im_id = train_ids[0]
im_pil = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
im_np = np.array(im_pil, dtype=np.uint8)
print(im_np.shape)

bbs = utils.get_bbs(df_summary, im_id)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(im_np)

for bb in bbs:
    utils.draw_bb(ax, bb)
# -

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(utils.blur_bbs(im_np, bbs))

# # Load pre-trained model

vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
vgg.eval()

modules = list(vgg.features.children())
feature_model = nn.Sequential(*modules)[:52]

# # Bounding box activation spread

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

def activation_spread(model, im, bbs):
    y1 = get_activations(model, im)
    upper = np.quantile(y1, q=0.9, axis=(1,2))
    lower = np.quantile(y1, q=0.1, axis=(1,2))
    bounds_spread = upper - lower
    
    y2 = get_activations(model, utils.blur_bbs(im, bbs))
    blur_spread = np.mean(np.abs(y1 - y2), axis=(1,2))
    
    spread = bounds_spread + blur_spread
    
    return spread


# +
n = 100
layer_wts = dict()
for i in range(10):
    layer_spreads = np.zeros(512)
    
    for im_id in np.random.choice(train_ids, n):
        im = Image.open(f'''{DATA_PATH}/train/{im_id}.jpg''')
        bbs = utils.get_bbs(df_summary, im_id)

        layer_spreads += activation_spread(feature_model, im, bbs)
    
    layer_spreads /= n
    layer_wts[f'wts{i}'] = layer_spreads/np.sum(layer_spreads)
    
np.savez(f'{DATA_PATH}/vgg19_feature_layer_wts.npz', **layer_wts)
# -
layer_wts = np.load(f'{DATA_PATH}/vgg19_feature_layer_wts.npz')
