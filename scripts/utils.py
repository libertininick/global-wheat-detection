import ast

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def get_bbs(df_summary, image_id):
    """Gets the bounding boxes for an image

    Args:
         df_summary (DataFrame)
         image_id (str)

    Returns:
        bbs (list)
    """
    bbs = [[int(x) for x in ast.literal_eval(bb)] 
          for bb 
          in df_summary.query(f'''image_id == '{image_id}' ''')['bbox']
          ]

    return bbs


def slice_to_bb(im, bb):
    j, i, w, h = bb
    return im[i:i+h, j:j+w]


def binarize(im):
    """Binarize an image into background (0) and foreground (1) using Otsu's method
    """
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    _, im_binary = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
    return im_binary


def remove_noise(im, kernel=np.ones((3,3), np.uint8), iterations=2):
    """Remove noise in a binary image via morphological transformations
    """
    im_denoise = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations)
    im_denoise = cv2.morphologyEx(im_denoise, cv2.MORPH_CLOSE, kernel, iterations)
    return im_denoise


def label_largest_component(im):
    """Labels largest connected foreground component
    """
    labels = np.zeros_like(im, np.uint8)

    # Run connected components on foreground
    _, components = cv2.connectedComponents(im)
    comp_ids, comp_counts = np.unique(components, return_counts=True)
    
    if len(comp_ids) > 1:
        biggest_comp = comp_ids[1:][np.argmax(comp_counts[1:])]
        labels[components == biggest_comp] = 1

    return labels


def bbs_to_segmentation_mask(im, bbs):
    """Converts instance bounding boxes in an image to a single segmentation mask

    Args:
        im (ndarray): Numpy array of image (colors, height, width)
    """

    # Initialize mask with all 0s
    mask = np.zeros(im.shape[:2])
    
    for bb in bbs:
        # Fill in portion of mask associated with bb segment
        j, i, w, h = bb
        
        im_slice = im[i:i+h, j:j+w]
        im_slice = binarize(im_slice)
        im_slice = remove_noise(im_slice)
        
        mask[i:i+h, j:j+w] += label_largest_component(im_slice)
    
    # Trunc overlapping segments
    mask = np.minimum(1, mask)
    
    return mask


def draw_bb(ax, bb):
    """Draws bounding box on an image
    """
    *cords, w, h = bb
    rect = patches.Rectangle(cords, w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


def blur_bbs(im, bbs):
    """Blur all regions inside bounding boxes
    """
    im = np.array(im, dtype=np.uint8)
    
    h, w, _ = im.shape
    window_h, window_w = int(((h*0.2)//2)*2 + 1), int(((w*0.2)//2)*2 + 1)
    blur = cv2.GaussianBlur(im, (window_h, window_w), 0)
    
    for bb in bbs:
        # Fill in portion of image with blurred image 
        j, i, w, h = bb
        im[i:i+h, j:j+w] = blur[i:i+h, j:j+w]

    smooth = cv2.GaussianBlur(im, (5, 5), 0)
    
    return Image.fromarray(np.uint8(smooth))