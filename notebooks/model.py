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

import global_wheat_detection.scripts.modules as modules
import global_wheat_detection.scripts.utils as utils
# -

down_sampler = modules.DownsampleBlock(in_channels=3, n_downsamples=2)

m = modules.DenseDilationNet(in_channels=48
                             , n_feature_maps=8
                             , block_size=6
                             , n_blocks=3
                             , kernel_sizes=[(3,3)]
                             , global_kernel_size=3
                            )

x = torch.randn(4, 3, 512, 512)
with torch.no_grad():
    y, p = down_sampler(x)
    y2, gfv = m(y)

y2.shape

1024*.2


