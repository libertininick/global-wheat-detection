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

m = modules.WheatHeadDetector()

x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    y1, y2, y3, y4, y5 = m._forward_train(x)

y1.shape


