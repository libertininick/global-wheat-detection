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

import global_wheat_detection.scripts.modules as modules
import global_wheat_detection.scripts.utils as utils
# -

# # Data

# +
DATA_PATH = 'C:/Users/liber/Dropbox/Python_Code/global_wheat_detection/data'

df_summary = pd.read_csv(f'{DATA_PATH}/train.csv')
image_ids = df_summary['image_id'].values

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






hd_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds1 = modules.DownsampleUnit(3, 8)
ds2 = modules.DownsampleUnit(8, 24)

model = modules.DenseDilationNet(block_size=3
                         , in_channels=24
                         , n_feature_maps=4
                         , kernel_sizes=[(3,3)]
                         , n_global_feats=8
                         , global_kernel_size=(3,3)       
                         , swish_beta_scales=[0.6,0.8,1]
                         , dilations=[1,2,4]
                         , interleave_dilation=True
                         , dropout=0.1
                        )

ap = nn.AdaptiveMaxPool2d(output_size=(14,14))
channel_exp = nn.Conv2d(in_channels=68, out_channels=512, kernel_size=1)

with torch.no_grad():
    x = hd_preprocess(im_pil)
    x = x.unsqueeze(0)
    x = ds1(x)
    x = ds2(x)
    x = model(x)
    yh = channel_exp(ap(x))

#layer_wts = torch.from_numpy(layer_wts)
layer_wts = layer_wts.view(1, 512, 1, 1)

error = (yh - y1)**2*layer_wts
torch.mean(torch.sum(error, dim=1))



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
