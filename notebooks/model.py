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

DATA_PATH = os.path.abspath('../data')
loader = DataLoader(path=DATA_PATH, seed=123)

m = modules.WheatHeadDetector()

x, *y, bboxes_aug = loader.load_batch(batch_size=4, resolution_out=256)
x.shape

y[1].dtype

yh = m._forward_train(x)
training_utils.training_loss(*yh, *y)

yh = m._forward_inference(x)
bboxes_pred = training_utils.inference_output(yh, *list(x.shape[2:]))
utils.mAP(bboxes_pred, bboxes_aug)

# # Test training

model = modules.WheatHeadDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(10):
    x, *y, bboxes_aug = loader.load_batch(batch_size=4, resolution_out=256)
    yh = model._forward_train(x)

    loss = training_utils.training_loss(*yh, *y)
    print(loss.item())
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


