import itertools
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import scipy
import torch
import torch.nn as nn
from torchvision import transforms

import global_wheat_detection.scripts.utils as utils


#region Crop
def center_crop(im, h, w):
    """Center crop an image to (h, w).
    """
    
    h_o, w_o = im.shape[:2]
    c_x, c_y = w_o//2, h_o//2
    h_half, w_half = h//2, w//2
    
    im_cropped = im[ max(0, c_y - h_half):min(h_o, c_y + h_half)
                   , max(0, c_x - w_half):min(w_o, c_x + w_half)
                   ]
    
    return im_cropped


def crop(im, x, y, w, h):
    """Crop image

    Args:
        im (ndarray): Numpy image
        x (int): Left x coordinate of cropped image 
        y (int): Top y coordinate of cropped image 
        h  (int): height of cropped image
        w  (int): width of cropped image

    Returns:
        im_cropped (ndarray)
    """

    return im[y:y+h, x:x+w]
    

def crop_box(bboxes, x, y, w, h, alpha=0.2):
    """Crop the bounding boxes to the borders of an image
    
    Args:
        bboxes (list | ndarray): Numpy array containing bounding boxes of shape `N X 4` 
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
    bboxes = np.array(bboxes)

    bb_x1, bb_y1 = bboxes[:,0], bboxes[:,1]
    bb_w, bb_h = bboxes[:,2], bboxes[:,3]
    bb_areas = bb_w*bb_h + 1e-6

    # Crop
    bb_x4, bb_y4 = np.minimum(bb_x1 + bb_w, x + w), np.minimum(bb_y1 + bb_h, y + h)
    bb_x1, bb_y1 = np.maximum(bb_x1, x), np.maximum(bb_y1, y)
    bb_w, bb_h = np.maximum(0, bb_x4 - bb_x1), np.maximum(0, bb_y4 - bb_y1)
    bboxes_cropped = np.stack((bb_x1 - x, bb_y1 - y, bb_w, bb_h), axis=1) # shift by x, y

    # Area filter
    area_ratios = bb_w*bb_h/bb_areas
    mask = area_ratios > alpha
    
    bboxes_cropped = bboxes_cropped[mask, :]

    return bboxes_cropped
#endregion


#region Resize 
def scale_shape(h, w, scale_percent):
    """Resizes an image's shape based on a scaling percentage between 0% and inf

    Args:
         h (int): Image height
         w (int): Image width
         scale_percent (float): Scaling percentage [0, inf)

    Returns:
        scaled_shape (tuple): Height, width, channels
    """

    h, w = int(h*scale_percent), int(w*scale_percent)
    
    return h, w


def resize_im(im, scale_percent):
    """Resize an image

    Resizes an image based on a scaling percentage between 0% and inf

    Args:
         im (ndarray): Numpy image
         scale_percent (float): Scaling percentage [0, inf)

    Returns:
        im_resized (ndarray): Numpy image
    """
    h, w = scale_shape(*im.shape[:2], scale_percent)
    im_resized = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    
    return im_resized


def resize_bboxes(bboxes, h_o, w_o, h, w):
    bbox_norms = utils.normalize_bboxes(bboxes, h_o, w_o)
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
    h, w = im.shape[:2]
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
#endregion


#region Color augmentation
def _adjust_brightness(im, value):

    if value > 0:
        shadow = value
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + value
    
    alpha = (highlight - shadow)/255
    gamma = shadow

    return cv2.addWeighted(im, alpha, im, 0, gamma)


def _adjust_contrast(im, value):
    alpha = 131*(value + 127)/(127*(131 - value))
    gamma = 127*(1 - alpha)
    
    return cv2.addWeighted(im, alpha, im, 0, gamma)


def adjust_color(im, brightness=0, contrast=0, hue=0, saturation=0):
    """Adjust brightness, contracts, hue, and saturation of an image.
    
    Args:
        im (ndarray): Numpy image
        brightess (float): Brigtness adjustment (-1.0 to +1.0)
        contrast (float): Contrast adjustment (-1.0 to +1.0)
        hue (float): Hue adjustment (-1.0 to +1.0)
        saturation (float): Saturation adjustment (-1.0 to +1.0)
        
    Returns:
        im_adjusted (ndarray): Numpy image
    """
    
    # Scale adjustment values
    brightness = 255*min(max(-1, brightness), 1)
    contrast = 127*min(max(-1, contrast), 1)
    hue = np.uint8(255*min(max(-1, hue), 1))
    saturation = np.uint8(255*min(max(-1, saturation), 1))

    # RGB to BGR
    im_adjusted = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    
    # Bightness and contrast
    im_adjusted = _adjust_brightness(im_adjusted, brightness)
    im_adjusted = _adjust_contrast(im_adjusted, contrast)

    # BGR to HSV
    im_adjusted = cv2.cvtColor(im_adjusted, cv2.COLOR_BGR2HSV)

    # Hue and saturation
    im_adjusted[:,:,0] += hue
    im_adjusted[:,:,1] += saturation
    
    # HSV to RGB
    im_adjusted = cv2.cvtColor(im_adjusted, cv2.COLOR_HSV2RGB)

    return im_adjusted
#endregion


#region Blur/sharpen
def blur(im, blur_p):
    """
    Args:
        im (ndarray): Numpy image
        blur_p (float): Blurring percentage adjustment.
                        Reasonable values are between 0 - 5%
    """

    h, w = im.shape[:2]
    kernel_size = int(min(h, w)*blur_p//2)*2 + 1
    im_blurred = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)

    return im_blurred


def sharpen(im):
    """Return a sharpened version of the image

    Args:
        im (ndarray): Numpy image
    """

    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    
    im_sharpened = cv2.filter2D(im, -1, kernel)
    
    return im_sharpened
#endregion


class DataAugmentor():
    def __init__(self, seed=None):
        self.rnd = np.random.RandomState(seed)

    def _random_crop_dims(self, h_o, w_o, h, w):
        """Get valid x, y dimensions for cropping image to (h, w)"""
    
        h = min(h_o, h)
        w = min(w_o, w)

        x = self.rnd.choice(np.arange(w_o - w + 1), size=1).item(0)
        y = self.rnd.choice(np.arange(h_o - h + 1), size=1).item(0)

        return x, y, w, h

    def _random_puzzle_dims(self, h, w):

        w_left = int(w*self.rnd.random())
        w_right = w - w_left

        h_top = int(h*self.rnd.random())
        h_bottom = h - h_top

        puzzle_dims = [ (0, 0, w_left, h_top)
                      , (w_left, 0, w_right, h_top)
                      , (0, h_top, w_left, h_bottom)
                      , (w_left, h_top, w_right, h_bottom)
                      ]

        return np.array(puzzle_dims)

    def random_crop_resize(self, im, seg_mask, bboxes):
        """Randomly crop and then resize to original size an image, 
        its segmentation mask, and its bounding boxes
        """

        # Cropping dimensions
        h_o, w_o = im.shape[:2]
        a, b = self.rnd.rand(2)**0.1
        h, w = int(h_o*a), int(w_o*b)
        x, y, w, h = self._random_crop_dims(h_o, w_o, h, w)

        # Crop
        im_cropped = crop(im, x, y, w, h)
        mask_cropped = crop(seg_mask, x, y, w, h)
        bbs_cropped = crop_box(bboxes, x, y, w, h)

        # Resize
        im_resized = cv2.resize(im_cropped, (w_o, h_o), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_cropped, (w_o, h_o), interpolation=cv2.INTER_AREA)
        bbs_resized = resize_bboxes(bbs_cropped, *im_cropped.shape[:2], *im.shape[:2])

        return im_resized, mask_resized, bbs_resized

    def rotate(self, im, seg_mask, bboxes, angle, alpha=0.2):
        """Rotate an image and its bounding boxes
        """

        # Resize pre-rotation
        scale = 1 + np.abs(np.sin(2*(angle/360*2*np.pi))/2)
        im_resized = resize_im(im, scale)
        mask_resized = resize_im(seg_mask, scale)
        bboxes_resized = resize_bboxes(bboxes, *im.shape[:2], *im_resized.shape[:2])

        # Rotate
        im_rotated = rotate_im(im_resized, angle)
        mask_rotated = rotate_im(mask_resized, angle)
        h, w = im_resized.shape[:2]
        cx, cy = w//2, h//2
        corners = get_corners(np.array(bboxes_resized))
        corners = rotate_corners(corners, angle, cx, cy, h, w)
        bboxes_rotated = get_enclosing_box(corners)

        # Crop out black
        im_cropped = center_crop(im_rotated, *im.shape[:2])
        mask_cropped = center_crop(mask_rotated, *im.shape[:2])
        h, w = im_rotated.shape[:2]
        h_cropped, w_cropped = im_cropped.shape[:2]
        bboxes_cropped = crop_box( bboxes_rotated
                                , (w - w_cropped)//2, (h - h_cropped)//2, w_cropped, h_cropped
                                , alpha
                                )
        
        return im_cropped, mask_cropped, bboxes_cropped

    def augment_image(self, im, seg_mask, bboxes, scale_percent=1):
        """Apply augmentation pipeline to a single image and its bounding boxes
        """
        im_aug, mask_aug, bbs_aug = im.copy(), seg_mask.copy(), bboxes.copy()

        # Augment colors
        if self.rnd.rand() < 0.5:
            b, c, h, s = np.tanh(self.rnd.randn(4)*0.1)
            im_aug = adjust_color(im, b, c, h, s)

        # Blur
        if self.rnd.rand() < 0.5:
            blur_p = self.rnd.rand()**2*0.03
            im_aug = blur(im_aug, blur_p)

        # Rotate
        if self.rnd.rand() < 0.5:
            angle = self.rnd.randint(low=0, high=360)
            im_aug, mask_aug, bbs_aug = self.rotate(im_aug, mask_aug, bbs_aug, angle)

        # Crop & Resize
        if self.rnd.rand() < 0.5:
            im_aug, mask_aug, bbs_aug = self.random_crop_resize(im_aug, mask_aug, bbs_aug)

        # Change resolution
        h_o, w_o = im_aug.shape[:2]
        im_aug = resize_im(im_aug, scale_percent)
        mask_aug = resize_im(mask_aug, scale_percent)
        bbs_aug = resize_bboxes(bbs_aug, h_o, w_o, *im_aug.shape[:2])

        return im_aug, mask_aug, bbs_aug

    def random_puzzles(self, ims, seg_masks, bboxes):
        
        n = len(ims)
        
        # Sample four random puzzle pieces
        puzzle_dims = self._random_puzzle_dims(*ims[0].shape[:2])
        
        # Shuffle images
        shuffled_idxs = self.rnd.choice(np.arange(n), n, replace=False).tolist()

        im_puzzles, mask_puzzles, bb_puzzles = [], [], []

        for _ in range(n):

            im_pieces, mask_pieces, bb_pieces = [], [], []
            for j, (idx, (x, y, w, h)) in enumerate(zip(shuffled_idxs, puzzle_dims)):
                
                im_pieces.append(crop(ims[idx], x, y, w, h))
                mask_pieces.append(crop(seg_masks[idx], x, y, w, h))
                
                bbs = crop_box(bboxes[idx], x, y, w, h)
                if j%2 == 1:
                    bbs[:, 0] = bbs[:, 0] + max(puzzle_dims[:, 0])
                if j >= 2:
                    bbs[:, 1] += np.max(puzzle_dims[:, 1])
                bb_pieces.append(bbs)

            # Splice together pieces
            im_top = np.hstack(im_pieces[:2])
            im_bottom = np.hstack(im_pieces[2:])
            im_puzzles.append(np.vstack((im_top, im_bottom)))

            mask_top = np.hstack(mask_pieces[:2])
            mask_bottom = np.hstack(mask_pieces[2:])
            mask_puzzles.append(np.vstack((mask_top, mask_bottom)))

            bb_puzzles.append(np.concatenate(bb_pieces))

            # Shift indexes
            shuffled_idxs = shuffled_idxs[1:] + shuffled_idxs[:1]

        return im_puzzles, mask_puzzles, bb_puzzles

    def augment_batch(self, ims, seg_masks, bboxes, scale_percent):

        # batch size
        batch_size = len(ims)

        augs = [self.augment_image(im, seg_mask, bbs, scale_percent) 
                for im, seg_mask, bbs 
                in zip(ims, seg_masks, bboxes)
               ]

        if self.rnd.rand() < 0.5 and batch_size%4 == 0:
            im_puzzles, mask_puzzles, bb_puzzles = [],[],[]
            for idx in range(0, len(augs), 4):
                
                i, m, b = self.random_puzzles(*zip(*augs[idx:idx+4]))
                im_puzzles.extend(i)
                mask_puzzles.extend(m)
                bb_puzzles.extend(b)

            return im_puzzles, mask_puzzles, bb_puzzles
        else:
            return zip(*augs)


class DataLoader():
    def __init__(self, path, n_downsamples=2, seed=None):
        
        self.augmentor = DataAugmentor()
        self.cropper =  transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        self.normalizer = transforms.Compose([ transforms.ToTensor()
                                             , transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                                   , std=[0.229, 0.224, 0.225]
                                                                   )
                                            ])

        # # Set device
        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda:0'

        # # Load pre-trained VGG
        # vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
        # vgg.eval()
        # modules = list(vgg.features.children())
        # self.feature_model = nn.Sequential(*modules)[:52]
        # self.feature_model.to(self.device) # Move model to GPU
        # self.layer_wts = np.load(f'{path}/vgg19_feature_layer_wts.npz')
        # self.wts_keys = list(self.layer_wts.keys())
        
        self.rnd = np.random.RandomState(seed)
        self.path = path
        
        self.df_summary = pd.read_csv(f'{path}/train.csv')

        image_ids = dict()
        for _, (im_id, *o, source) in self.df_summary.iterrows():
            image_ids[im_id] = source  

        self.train_ids = list(image_ids.keys())[:-373]
        self.valid_ids = list(image_ids.keys())[-373:]

        # Parameters for bounding box targets
        self.n_downsamples = n_downsamples

        # Adaptive bounding box spread
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(8,8))

    def _load_raw(self, batch_size, split):
        """Load images and bounding boxes from disk

        Args:
            batch_size (int)
            split (str): 'train' | 'validation' | 'test'

        Returns:
            ims (list): List of numpy images
            bboxes (list): list of bbox arrays
        """
        
        # Load from disk
        ims, bboxes = [],[]
        if split == 'test':
            path, _, files = next(os.walk(f'{self.path}/train'))
        else:
            path = f'{self.path}/train'
            if split == 'validation':
                files = [f'{im_id}.jpg' for im_id in self.valid_ids]
            else:
                files = [f'{im_id}.jpg' for im_id in self.train_ids]

        for f in self.rnd.choice(files, size=batch_size):
            im = Image.open(f'{path}/{f}')
            im = np.array(im, dtype=np.uint8)
            ims.append(im)
            
            if split != 'test':
                bboxes.append(utils.get_bboxes(self.df_summary, f.replace('.jpg', '')))

        return ims, bboxes

    def _load_n_augment(self, batch_size, resolution_out, split):
        """Load a batch of images and augment
        
        Args:
            batch_size (int): Number of images to load
            resolution_out (int): Number of pixels for H & W
            split (str): 'train' | 'validation' | 'test'
        
        Returns:
            ims_aug (list): List of numpy images
            masks_aug (list): list of segmentation heat map masks
            bboxes_aug (list): list of bbox arrays
        """

        # Load from disk
        ims, bboxes = self._load_raw(batch_size, split)
        h, w, _ = ims[0].shape

        # Build bounding box segmentation masks
        seg_masks = [utils.bbox_segmentation_mask(bbs, h, w) for bbs in bboxes]
        
        # Data augmentation
        if resolution_out is not None:
            scale_percent = resolution_out/max(ims[0].shape)
        else:
            scale_percent = 1
        ims_aug, masks_aug, bboxes_aug = self.augmentor.augment_batch(ims, seg_masks, bboxes, scale_percent)

        return ims_aug, masks_aug, bboxes_aug

    @torch.no_grad()
    def _pretained_targets(self, ims):
        x = torch.stack([self.normalizer(self.cropper(Image.fromarray(im))) for im in ims], dim=0)
        x = x.to(self.device)
        x = self.feature_model(x)
        x = x.to('cpu')

        if self.device != 'cpu':
            torch.cuda.empty_cache()

        # Sample layer weights
        wts_key = self.rnd.choice(self.wts_keys, size=1).item(0)
        wts = self.layer_wts[wts_key]
        wts /= np.mean(wts)
        wts = torch.from_numpy(wts).view(1,-1,1,1).type(torch.float32)
        x = x*wts

        return x

    def load_batch(self, batch_size, resolution_out=None, split='train'):
        """

        Args:
            batch_size (int): Number of images to load
            resolution_out (int): Resolution to scale to. 
                                  If `None` resolution will be same as loaded images.
            split (str): 'train' | 'validation' | 'test'

        Returns:
            x (tensor): Normalized input images [b, c, h, w]
            y_n_bboxes (tensor): Number of bounding boxes per image [b, 1]
            y_bbox_spread (tensor): Probability spread of bounding boxes count over 8x8 grid [b, 1, 8, 8]
            y_seg (tensor): Bounding box segmentation mask [b, 1, h_ds, w_ds]
            y_bboxes (tensor): Bounding box area/shape targets [b, 3, h_ds, w_ds]
            seg_wts (tensor): Weights for segmentation loss based on distance from bbox centroids [b, 1, h_ds, w_ds]
            ims (ndarray): Numpy images
            bboxes (ndarray): Ground truth bounding boxes
        """
        
        if split != 'test':
            ims, seg_masks, bboxes = self._load_n_augment(batch_size, resolution_out, split)
        else:
            ims, bboxes = self._load_raw(batch_size, split)

        # Input tensors
        x = torch.stack([self.normalizer(Image.fromarray(im)) for im in ims], dim=0)
        h, w = list(x.shape[-2:])
        
        if split != 'test':
            # Number of bounding boxes
            y_n_bboxes = np.array([len(bbs)**0.75 for bbs in bboxes], dtype=np.float32)
            y_n_bboxes = (y_n_bboxes - 16.65)/6.07  # Normalize based on training image stats
            

            # Segmentation
            h_ds, w_ds = h//(2**self.n_downsamples), w//(2**self.n_downsamples)
            seg_masks = [cv2.resize(mask, (w_ds, h_ds), interpolation=cv2.INTER_AREA) for mask in seg_masks]
            y_seg = np.stack(seg_masks, axis=0)
            y_seg[y_seg > 0] = 1.0
            y_seg[y_seg != 1] = 0.0

            # Bounding box targets
            targets = [utils.bbox_targets(bbs, h, w, self.n_downsamples) for bbs in bboxes]
            y_bboxes = np.stack(targets, axis=0)

            # Bounding box spread over an 8x8 grid
            y_bbox_spread = torch.from_numpy(y_bboxes[:, 0, :, :])  # Centroids
            y_bbox_spread = self.adaptive_pool(y_bbox_spread)
            y_bbox_spread = y_bbox_spread/(torch.sum(y_bbox_spread, dim=(1,2))[:,None,None])

            # Segmentation weights
            idxs = itertools.product(np.arange(h_ds), np.arange(w_ds))
            idxs = np.array(list(idxs))

            seg_wts = []
            for b in range(batch_size):
                centroid_idxs = np.array(np.where(y_bboxes[b, 0, :, :] == 1)).T
                wts = scipy.spatial.distance_matrix(idxs, centroid_idxs)
                wts = np.min(wts, axis=1)
                wts = np.argsort(np.argsort(-wts))
                wts = wts/np.sum(wts)
                wts = wts.reshape(h_ds,w_ds)
                seg_wts.append(wts)
            seg_wts = np.stack(seg_wts, axis=0)

            # To tensors
            y_n_bboxes = torch.from_numpy(y_n_bboxes).unsqueeze(1)
            y_bbox_spread = y_bbox_spread.unsqueeze(1)
            y_seg = torch.from_numpy(y_seg).unsqueeze(1)
            y_bboxes = torch.from_numpy(y_bboxes)
            seg_wts = torch.from_numpy(seg_wts)

            y = (y_n_bboxes, y_bbox_spread, y_seg, y_bboxes, seg_wts)
        else:
            y = None

        return x, y, (ims, bboxes)