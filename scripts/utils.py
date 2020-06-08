import ast

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import scipy.sparse


def load_sparse_matrix(path):
    sparse_matrix = scipy.sparse.load_npz(path)
    arr = sparse_matrix.todense()
    return arr


#region mAP
def iou(box_a, box_b):
    """Calculate IoU of two bouding boxes
    
    Args:
        box_a (list): [xmin, ymin, width, height]
        box_b (list): [xmin, ymin, width, height]
    
    Returns:
        float: value of the IoU for the two boxes.
    
    """
    # Box a
    x1_a, y1_a, w_a, h_a = box_a
    x2_a = x1_a + w_a
    y2_a = y1_a + h_a
    
    # Box b
    x1_b, y1_b, w_b, h_b = box_b
    x2_b = x1_b + w_b
    y2_b = y1_b + h_b
    
    # No overlap check
    if x2_a < x1_b:
        return 0
    elif x2_b < x1_a:
        return 0
    elif y2_a < y1_b:
        return 0
    elif y2_b < y1_a:
        return 0
    
    # Box areas
    area_a = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    area_b = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Intersection 
    x1_max = max(x1_a, x1_b)
    x2_min = min(x2_a, x2_b)
    y1_max = max(y1_a, y1_b)
    y2_min =min(y2_a, y2_b)
    area_i = (x2_min - x1_max + 1) * (y2_min - y1_max + 1)
    
    iou = area_i / (area_a + area_b - area_i)
    
    return iou


def iou_pairs(boxes_pred, boxes_true):
    """Calculates the IoU between every pair of predictions and ground truth objects
    
    Args:
        boxes_pred (ndarray): (n_predictions, 4)
        boxes_true (ndarray): (n_objects, 4)
        
    Returns:
        ious (ndarray): (n_predictions, n_objects)
        
    """
    
    ious = [[iou(bb_p, bb_t) for bb_t in boxes_true] for bb_p in boxes_pred]
    return np.array(ious)


def precision(ious, confidence_levels=None, iou_threshold=0.5):
    """Calculates the prediction precision for a specific IoU threshold
    
    Args:
        ious (ndarray): (n_predictions, n_objects)
        confidence_levels (list): Confidence level between 0 and 1 for each prediction
        iou_threshold (float): value of IoU to consider as threshold for a true prediction.
    
    Returns:
        precision (float): TP(t) / (TP(t) + FP(t) + FN(t))
    """

    # Order predictions by confidence (most -> least)
    if confidence_levels is not None: 
        ious = ious[np.argmax(-np.array(confidence_levels)), :]
        
    object_idxs = np.arange(ious.shape[1], dtype=np.uint8)
    assigned_objects = np.array([], dtype=np.uint8)
    
    # Iterate across predictions counting TPs and FPs
    tp, fp = 0, 0
    for predicition_ious in ious:
        
        unassigned_idxs = np.setdiff1d(object_idxs, assigned_objects)
        
        if len(unassigned_idxs) > 0:
            predicition_ious = predicition_ious[unassigned_idxs]
            iou_max = np.argmax(predicition_ious)
            
            if predicition_ious[iou_max] >= iou_threshold:
                tp += 1
                assigned_objects = np.append(assigned_objects, unassigned_idxs[iou_max])
            else:
                fp += 1
        else:
            fp += 1

    # Count left over objects and FNs
    fn = len(object_idxs) - len(assigned_objects)

    return tp/(tp + fp + fn)


def average_precision( boxes_pred
                      , boxes_true
                      , confidence_levels=None
                      , iou_thresholds=np.arange(50, 80, 5)/100
                     ):
    """Calculates the average prediction precision for a range IoU thresholds
    
    Args:
        boxes_pred (ndarray): (n_predictions, 4)
        boxes_true (ndarray): (n_objects, 4)
        confidence_levels (list): Confidence level between 0 and 1 for each prediction
    
    Returns:
        avg_precision (float)
    """
        
    ious = iou_pairs(boxes_pred, boxes_true)
    precisions = [precision(ious, confidence_levels, t) for t in iou_thresholds]
    avg_precision = np.mean(precisions)
    
    return avg_precision


def mAP(y_pred, y_true, iou_thresholds=np.arange(50, 80, 5)/100):
    """Calculates the mean average prediction precision for set of images
    
    Args:
        y_pred (list): List of predictions for each image
                       Each image should have an array of 
                       bounding box predicitions including 
                       prediction confidences
                       [confidence, xmin, ymin, width, height]
        y_true (list): List of bounding boxes for each image
                       [xmin, ymin, width, height]
        iou_thresholds (list): Range of IoU thresholds for average precision calc
        
    Returns:
        mean_avg_precision (float)
    """

    avg_precisions = [average_precision(y_p[:,1:], y_t, y_p[:,0], iou_thresholds) 
                      for y_p, y_t 
                      in zip(y_pred, y_true)
                     ]
    mean_avg_precision = np.mean(avg_precisions)
    
    return mean_avg_precision
#endregion


#region Bounding Boxes
def get_bboxes(df_summary, image_id):
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


def draw_bboxes(ax, bbox, color='r'):
    """Draws bounding box on an image
    """
    *cords, w, h = bbox
    rect = patches.Rectangle(cords, w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def blur_bboxes(im, bboxes):
    """Blur all regions inside bounding boxes
    """
    im = np.array(im, dtype=np.uint8)
    
    h, w, _ = im.shape
    window_h, window_w = int(((h*0.2)//2)*2 + 1), int(((w*0.2)//2)*2 + 1)
    blur = cv2.GaussianBlur(im, (window_h, window_w), 0)
    
    for bb in bboxes:
        # Fill in portion of image with blurred image 
        j, i, w, h = bb
        im[i:i+h, j:j+w] = blur[i:i+h, j:j+w]

    smooth = cv2.GaussianBlur(im, (5, 5), 0)
    
    return Image.fromarray(np.uint8(smooth))


def normalize_bboxes(bboxes, h, w):
    """Normalize bounding boxes to an image's height and width
    
    Args:
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
        h (int): Image height
        w (int): Image width

    Returns:
        bboxes (ndarray): (N, 4)
    """
    bbox_norms = np.array(bboxes)/np.array([w, h, w, h])
    
    return bbox_norms


def bbox_targets(bboxes, h, w, model_downsample=2, area_scale=10):
    """Builds centroid mask, area ratio regression mesh, and 
    side ratio regression mesh for a list of bounding boxes
    
    Args:
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
        h (int): Image height
        w (int): Image width
        model_downsample (int): Downs
        area_scale=10

    Returns:
        centroid_mask (ndarray): (h//model_downsample, w//model_downsample)
        area_ratios_mesh (ndarray): (h//model_downsample, w//model_downsample)
        side_ratios_mesh (ndarray): (h//model_downsample, w//model_downsample)
    """
    # Factor in downsampling of image as it passes through model
    h_ds, w_ds = h//model_downsample, w//model_downsample

    # Normalize bounding boxes for original height and width of image
    bboxes_norm = normalize_bboxes(bboxes, h, w)
    
    # Centroid mask for NLLLoss 
    centroid_mask = np.zeros((h_ds, w_ds), dtype=np.int64)
    centroid_idxs = bboxes_norm[:, :2]
    centroid_idxs[:,0] += bboxes_norm[:, 2]/2
    centroid_idxs[:,1] += bboxes_norm[:, 3]/2
    centroid_idxs = np.floor(centroid_idxs*np.array([h_ds, w_ds])).astype(np.uint8)
    centroid_mask[centroid_idxs[:,0], centroid_idxs[:,1]] = 1

    # Area ratios mesh for MSE
    area_ratios = np.prod(bboxes_norm[:,-2:], axis=1)**(1/area_scale)
    area_ratios_mesh = np.zeros((h_ds, w_ds), dtype=np.float32)
    area_ratios_mesh[centroid_idxs[:,0], centroid_idxs[:,1]] = area_ratios

    # Side ratios mesh for MSE 
    side_ratios = bboxes_norm[:,2]/np.sum(bboxes_norm[:,-2:], axis=1)
    side_ratios_mesh = np.zeros((h_ds, w_ds), dtype=np.float32)
    side_ratios_mesh[centroid_idxs[:,0], centroid_idxs[:,1]] = side_ratios

    return centroid_mask, area_ratios_mesh, side_ratios_mesh


def bbox_pred_to_dims(area_ratio, side_ratio, w, h, area_scale=10):
    """Converts sigmoid prediction in bbox w, h dims

    Args:
        area_ratio (float): ((bb_w*bb_h)/(im_w*im_h))**(1/area_scale)
        side_ratio (float): bb_w/(bb_w + bb_h)
        w (int): Image width
        h (int): Image height
        area_scale (int): Root used to scale area ratio

    Returns:
        bb_w_pred (int)
        bb_h_pred (int)
    """
    area_pred = area_ratio**area_scale*w*h
    coeff = [1 - side_ratio, 0, -side_ratio*area_pred]
    w_pred = int(np.round(max(np.roots(coeff))))
    h_pred = int(np.round(area_pred/w_pred))
    
    return w_pred, h_pred
#endregion


#region Segmentation heat map
def binarize(im):
    """Binarize an image into background (0) and foreground (1) using Otsu's method
    """
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    _, im_binary = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
    return im_binary/255


def remove_noise(im, kernel=np.ones((3,3), np.uint8), iterations=2):
    """Remove noise in a binary image via morphological transformations
    """
    im_denoise = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations)
    im_denoise = cv2.morphologyEx(im_denoise, cv2.MORPH_CLOSE, kernel, iterations)
    return im_denoise


def segmentation_heat_map(im, bboxes, kernel=np.ones((5,5), dtype=np.uint8)):
    """Converts instance bounding boxes in an image to a single segmentation mask

    Args:
        im (ndarray): Numpy array of image (colors, height, width)
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
    """

    # Initialize mask with all 0s
    mask = np.zeros(im.shape[:2], dtype=np.float32)
    
    for bb in bboxes:
        # Fill in portion of mask associated with bb segment
        x, y, w, h = bb
        
        im_slice = im[y:y+h, x:x+w]
        im_slice = binarize(im_slice)
        im_slice = remove_noise(im_slice)

        im_dilated = cv2.dilate(im_slice, kernel, iterations=2)
        im_eroded = cv2.erode(im_slice, kernel, iterations=2)

        mask_slice = (im_dilated + im_slice + im_eroded)/3

        mask[y:y+h, x:x+w] += mask_slice

    return mask
#endregion

