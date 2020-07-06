import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from global_wheat_detection.scripts.preprocessing import DataLoader
import global_wheat_detection.scripts.utils as utils

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss(reduction='none')
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
log_softmax = torch.nn.LogSoftmax(dim=-1)
softmax = torch.nn.Softmax(dim=-1)

def training_loss( yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes
                 , y_n_bboxes, y_bbox_spread, y_seg, y_bboxes, seg_wts
                 ):
    """ Combined training loss across training objectives

        (1): Regress on number of bounding boxes with MSE loss
        (2): KL Divergence of bounding box spread
        (3): Bounding box segmentation mask classification BCE loss
        (4): Bounding box prediction
            (a): Bound box area ratios regression with MSE loss
            (b): Bounding box side ratios regression with MSE loss

    Args:
        yh_n_bboxes (tensor):    Predicted - Number of bounding boxes per image [b, 1]
        yh_bbox_spread (tensor): Predicted - Log-probability spread of bounding boxes count over 8x8 grid [b, 1, 8, 8]
        yh_seg (tensor):         Predicted - Bounding box segmentation mask [b, 1, h_ds, w_ds]
        yh_bboxes (tensor):      Predicted - Bounding box area/shape targets [b, 2, h_ds, w_ds]
        y_n_bboxes (tensor):     Target - Number of bounding boxes per image [b, 1]
        y_bbox_spread (tensor):  Target - Probability spread of bounding boxes count over 8x8 grid [b, 1, 8, 8]
        y_seg (tensor):          Tagret - Bounding box segmentation mask [b, 1, h_ds, w_ds]
        y_bboxes (tensor):       Target - Bounding box area/shape targets [b, 3, h_ds, w_ds]
        seg_wts (tensor):        Weights for segmentation loss based on distance from bbox centroids [b, 1, h_ds, w_ds]
    """

    loss_n_bboxes = mse_loss(yh_n_bboxes, y_n_bboxes)

    # b, c, h, w = list(yh_bbox_spread.shape)
    # yh_bbox_spread = log_softmax(yh_bbox_spread.view(b,c,-1)).view(b,c,h,w)
    # loss_bbox_spread = kl_loss(yh_bbox_spread, y_bbox_spread)
    
    loss_segmentation = bce_loss(yh_seg, y_seg).squeeze(1)
    loss_segmentation = loss_segmentation*seg_wts
    loss_segmentation = torch.sum(loss_segmentation, dim=(1,2))
    loss_segmentation = torch.mean(loss_segmentation)
    
    b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
    loss_bb_regressors, denom = 0, 2
    if len(i) > 0:
        loss_bb_regressors = mse_loss(yh_bboxes[b, :, i, j], y_bboxes[b, 1:, i, j])
        denom = 3
    
    return loss_n_bboxes, loss_segmentation*3, loss_bb_regressors, denom


def cluster_centroids(yh_seg, n_bboxes, threshold=0.5, kernel=np.ones((5,5), np.uint8), n_init=10):

    # Threshold segmentation predictions
    yh = (yh_seg >= threshold).astype(np.uint8)

    # Morphological smoothing on threshold predictions
    yh_smooth = cv2.morphologyEx(yh, cv2.MORPH_OPEN, kernel)
    yh_smooth = cv2.morphologyEx(yh_smooth, cv2.MORPH_CLOSE, kernel)

    # Positional extraction 
    i,j = np.where(yh_smooth == 1)
    yh_pos = np.stack((i,j), axis=-1)

    # Clustering on positional data
    kmeans = KMeans(n_clusters=n_bboxes, n_init=n_init, random_state=0).fit(yh_pos)
    
    # Extract centroids
    yh_centroids = np.round(kmeans.cluster_centers_).astype(np.int64)
    
    return yh_centroids


def inference_output(yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes, w, h):
    
    # Number of predicted bounding boxes
    n_bboxes = np.round((yh_n_bboxes.numpy().squeeze()*6.07 + 16.65)**(1.33), 0).astype(np.uint16)
    
    # # Bounding box spread
    # b, c, h, w = list(yh_bbox_spread.shape)
    # yh_bbox_spread = softmax(yh_bbox_spread.view(b,c,-1)).view(b,c,h,w)

    # Segmentation mask
    yh_seg = torch.sigmoid(yh_seg[:,0,:,:])
    yh_seg = yh_seg.numpy()
    b, h_ds, w_ds = yh_seg.shape  # Output shape

    yh_bboxes = yh_bboxes.numpy()
    
    bboxes = []
    # [confidence, xmin, ymin, width, height]
    for im_idx in range(b):
        n_bbs = n_bboxes[im_idx]

        # Bounding box centroids
        centroids = cluster_centroids(yh_seg[im_idx], n_bbs)
        i, j = centroids[:, 0], centroids[:, 1]
        
        # Confidence
        confidences = yh_seg[im_idx][i,j]

        # Normalize centroids
        centroids_norm = centroids/np.array([h_ds, w_ds])
        xs = centroids_norm[:, 1]
        ys = centroids_norm[:, 0]

        # Area and side predictions @ centroids
        areas = yh_bboxes[im_idx, 0, i, j]
        sides = yh_bboxes[im_idx, 1, i, j]

        bboxes.append(np.array([[c] + utils.bbox_pred_to_dims(x, y, a, s, w, h) 
                                for (c, x, y, a, s)
                                in zip(confidences, xs, ys, areas, sides)
                               ]))
            
    return bboxes


def cyclic_lr_scales(n_epochs, n_warmup, t=10, mult=2, max_t=160):
    """
    SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
        https://arxiv.org/pdf/1608.03983.pdf
    
    SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE 
        https://arxiv.org/pdf/1704.00109.pdf
    
    Bag of Tricks for Image Classification with Convolutional Neural Networks 
        https://arxiv.org/pdf/1812.01187.pdf
    """
    
    t_cur_schedule = []
    t_i_schedule = []
    while len(t_i_schedule) < n_epochs:
        t_cur_schedule.extend(range(t))
        t_i_schedule.extend([t]*t)
        t = min(max_t, t*mult)
        
    t_cur_schedule, t_i_schedule = t_cur_schedule[:n_epochs], t_i_schedule[:n_epochs]
    
    lr_scales = [0.5*(1 + np.cos(t_cur/t_i*np.pi)) for t_cur, t_i in zip(t_cur_schedule, t_i_schedule)]
    warmup_scales = np.linspace(0, 1, n_warmup + 1) # Add 1 to n_warmup for initial lr 
    
    return np.append(warmup_scales, lr_scales)


def train(model, data_path, save_path, n_epochs, n_warmup=5, n_steps_per_epoch=100, seed=123):
    loader = DataLoader(data_path, seed=seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

    lr_sales = cyclic_lr_scales(n_epochs, n_warmup)

    def lr_lambda(epoch):
        """
        'We use a mini batch size of 64'
        https://arxiv.org/pdf/1704.00109.pdf
        """
        return lr_sales[epoch]/64
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rnd = np.random.RandomState(seed=seed)

    res_batch_combos = [(512, 2), (400, 2), (300, 4), (256, 4)]

    losses = []
    for e_i in range(n_epochs):
        e_losses = []
        for s_i in range(n_steps_per_epoch):
            resolution_out, batch_size = rnd.choice(res_batch_combos)

            x, *y, bboxes_aug = loader.load_batch(batch_size, resolution_out, split='train')
            yh = model._forward_train(x)
            loss = training_loss(*yh, *y)

            loss.backward()
            e_losses.append(loss.item())
            
            lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr*batch_size # adj LR for batch size
            optimizer.step()
            optimizer.param_groups[0]['lr'] = lr

            optimizer.zero_grad()

        losses.append((np.median(e_losses), np.min(e_losses), np.max(e_losses)))
        
        scheduler.step()

        valid_mAP, valid_ct = 0, 0
        for _ in range(10):
            resolution_out, batch_size = rnd.choice(res_batch_combos)

            x, *y, bboxes_aug = loader.load_batch(batch_size, resolution_out, split='validation')
            yh = model._forward_inference(x)
            bboxes_pred = inference_output(yh, *list(x.shape[2:]))
            valid_mAP += utils.mAP(bboxes_pred, bboxes_aug)*batch_size
            valid_ct += batch_size

        #TODO Print progress
        #TODO Save model if on cyclic reset

        

