import cv2
import numpy as np
from PIL import Image


def normalize_bboxes(bboxes, im_shape):
    """Normalize bounding boxes to an images height and width"""
    h, w, _ = im_shape
    bbox_norms = np.array(bboxes)/np.array([w, h, w, h])
    
    return bbox_norms

def get_corners(bboxes):
    """Get corners of bounding boxes
    
    Args:
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
    
    Returns:
        corners (ndarray): Numpy array of shape `N x 8` containing N bounding boxes each
                           described by its corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    w, h = bboxes[:,2], bboxes[:,3]
    x1, y1 = bboxes[:,0], bboxes[:,1]
    x2, y2 = x1 + w, y1
    x3, y3 = x1, y1 + h
    x4, y4 = x1 + w, y1 + h
    
    corners = np.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)

    return corners

#region Crop
def center_crop(im, h, w):
    """Center crop an image to (h, w).
    """
    
    h_o, w_o, _ = im.shape
    c_x, c_y = w_o//2, h_o//2
    h_half, w_half = h//2, w//2
    
    im_cropped = im[ max(0, c_y - h_half):min(h_o, c_y + h_half)
                   , max(0, c_x - w_half):min(w_o, c_x + w_half)
                   ]
    
    return im_cropped


def crop_box(bboxes, x, y, w, h, alpha=0.01):
    """Crop the bounding boxes to the borders of an image
    
    Args:
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
        x (int): Left x coordinate of cropped image 
        y (int): Top y coordinate of cropped image 
        h  (int): height of cropped image
        w  (int): width of cropped image
        alpha (float): If the fraction of a bounding box left in the image after being cropped 
                       is less than `alpha` the bounding box is dropped. 
    
    Returns:
        bboxes_cropped (ndarray)
    
    """
    bb_x1, bb_y1 = bboxes[:,0], bboxes[:,1]
    bb_w, bb_h = bboxes[:,2], bboxes[:,3]
    bb_areas = bb_w*bb_h

    # Crop
    bb_x4, bb_y4 = np.minimum(bb_x1 + bb_w, x + w), np.minimum(bb_y1 + bb_h, y + h)
    bb_x1, bb_y1 = np.maximum(bb_x1, x), np.maximum(bb_y1, y)
    bb_w, bb_h = (bb_x4 - bb_x1), (bb_y4 - bb_y1)
    bboxes_cropped = np.stack((bb_x1 - x, bb_y1 - y, bb_w, bb_h), axis=1) # shift by x, y

    # Area filter
    area_ratios = bb_w*bb_h/bb_areas
    mask = area_ratios > alpha
    
    bboxes_cropped = bboxes_cropped[mask, :]

    return bboxes_cropped
#endregion


#region Resize 
def scale_shape(im_shape, scale_percent):
    """Resizes an image's shape based on a scaling percentage between 0% and inf

    Args:
         im_shape (tuple): Height, width, channels from `np.shape`
         scale_percent (float): Scaling percentage [0, inf)

    Returns:
        scaled_shape (tuple): Height, width, channels
    """
    h, w, c = im_shape
    h, w = int(h*scale_percent), int(w*scale_percent)
    
    return h, w, c


def resize_im(im, scale_percent):
    """Resize an image

    Resizes an image based on a scaling percentage between 0% and inf

    Args:
         im (ndarray): Numpy image
         scale_percent (float): Scaling percentage [0, inf)

    Returns:
        im_resized (ndarray): Numpy image
    """
    h, w, _ = scale_shape(im.shape, scale_percent)
    im_resized = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    
    return im_resized


def resize_bboxes(bboxes, orig_shape, new_shape):
    
    h, w, _ = new_shape
    bbox_norms = normalize_bboxes(bboxes, orig_shape)
    bbox_scaled = np.round(bbox_norms*np.array([w, h, w, h]))
    
    return bbox_scaled

#endregion


#region Rotation
def rotate_im(im, angle):
    """Rotate an image
    
    Rotate an image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Args:
        image (ndarray): Numpy image
        angle (float): Rotation angle in degrees. Positive values mean counter-clockwise rotation 
                       (the coordinate origin is assumed to be the top-left corner)
    
    Returns:
        im_rotated (ndarray): Numpy image
    
    """

    # Center of image
    h, w, _ = im.shape
    cx, cy = w//2, h//2

    # Transformation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    sin, cos = np.abs(M[0, 1]), np.abs(M[0, 0])

    # New bounding dimensions of the image
    h, w = int((h*cos) + (w*sin)), int((h*sin) + (w*cos))

    # Adjust the rotation matrix to take into account translation of center
    M[1, 2] += h/2 - cy
    M[0, 2] += w/2 - cx

    # Rotate the image
    im_rotated= cv2.warpAffine(im, M, (w, h))

    return im_rotated


def rotate_corners(corners, angle, cx, cy, h, w):
    
    """Rotate bounding box corners.
    
    Args:
        corners (ndarray): Numpy array of shape `N x 8` containing N bounding boxes each
                           described by its corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4` 
        angle (float): Rotation angle in degrees. Positive values mean counter-clockwise rotation 
                       (the coordinate origin is assumed to be the top-left corner)
        cx (int): x coordinate of the center of image (about which the box will be rotated)
        cy (int): y coordinate of the center of image (about which the box will be rotated)
        h  (int): height of the image
        w  (int): width of the image
    
    Returns:
        corners_rotated (ndarray): Numpy array of shape `N x 8` containing N rotated bounding boxes
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype=type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    sin, cos = np.abs(M[0, 1]), np.abs(M[0, 0])
    
    # New bounding dimensions of the image
    h, w = int((h*cos) + (w*sin)), int((h*sin) + (w*cos))
    
    # Adjust the rotation matrix to take into account translation of center
    M[1, 2] += h/2 - cy
    M[0, 2] += w/2 - cx
    
    # Prepare the vector to be transformed
    corners = np.dot(M, corners.T).T
    
    return corners.reshape(-1,8)


def get_enclosing_box(corners):
    """Get an enclosing box for rotated corners of a bounding box
    
    Args:
        corners (ndarray): Numpy array of shape `N x 8` containing N bounding boxes each
                           described by its corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4` 
    
    Returns:
        bboxes (ndarray): Numpy array containing bounding boxes of shape `N X 4` 
                          where N is the number of bounding boxes and the boxes 
                          are represented in the format `x, y, w, h`
        
    """
    x = corners[:,[0, 2, 4, 6]]
    y = corners[:,[1, 3, 5, 7]]
    
    x_min = np.min(x, axis=1)
    y_min = np.min(y, axis=1)
    x_max = np.max(x, axis=1)
    y_max = np.max(y, axis=1)

    w = x_max - x_min
    h = y_max - y_min
    
    bboxes = np.stack((x_min, y_min, w, h), axis=1)
    
    return bboxes

def rotate(im, bboxes, angle, alpha=0.2):
    """Rotate an image and its bounding boxes
    """

    # Resize pre-rotation
    scale = 1 + np.abs(np.sin(2*(angle/360*2*np.pi))/2)
    im_resized = resize_im(im, scale)
    bboxes_resized = resize_bboxes(bboxes, im.shape, im_resized.shape)

    # Rotate
    im_rotated = rotate_im(im_resized, angle)
    w, h, _ = im_resized.shape
    cx, cy = w//2, h//2
    corners = get_corners(np.array(bboxes_resized))
    corners = rotate_corners(corners, angle, cx, cy, h, w)
    bboxes_rotated = get_enclosing_box(corners)

    # Crop out black
    im_cropped = center_crop(im_rotated, *im.shape[:2])
    w, h, _ = im_rotated.shape
    w_cropped, h_cropped, _ = im_cropped.shape
    bboxes_cropped = crop_box( bboxes_rotated
                             , (w - w_cropped)//2, (h - h_cropped)//2, w_cropped, h_cropped
                             , alpha
                             )
    
    return im_cropped, bboxes_cropped
#endregion