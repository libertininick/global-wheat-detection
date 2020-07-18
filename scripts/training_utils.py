import numpy as np
import torch
import torch.nn as nn


mse_loss = nn.MSELoss()
seg_bce_loss = nn.BCEWithLogitsLoss()
centroid_bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([250.0]))
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
log_softmax = torch.nn.LogSoftmax(dim=-1)
softmax = torch.nn.Softmax(dim=-1)


def training_loss( yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes
                 , y_n_bboxes, y_bbox_spread, y_seg, y_bboxes
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
        yh_bboxes (tensor):      Predicted - Bounding box area/shape targets [b, 3, h_ds, w_ds]
        y_n_bboxes (tensor):     Target - Number of bounding boxes per image [b, 1]
        y_bbox_spread (tensor):  Target - Probability spread of bounding boxes count over 8x8 grid [b, 1, 8, 8]
        y_seg (tensor):          Tagret - Bounding box segmentation mask [b, 1, h_ds, w_ds]
        y_bboxes (tensor):       Target - Bounding box area/shape targets [b, 3, h_ds, w_ds]
    """

    loss_n_bboxes = mse_loss(yh_n_bboxes, y_n_bboxes)

    b, c, h, w = list(yh_bbox_spread.shape)
    yh_bbox_spread = log_softmax(yh_bbox_spread.view(b,c,-1)).view(b,c,h,w)
    loss_bbox_spread = kl_loss(yh_bbox_spread, y_bbox_spread)
    
    loss_segmentation = seg_bce_loss(yh_seg, y_seg)
    
    b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
    loss_bb_centroids, loss_bb_regressors, denom = 0, 0, 3
    if len(i) > 0:
        loss_bb_centroids = centroid_bce_loss(yh_bboxes[:, 0, :, :], y_bboxes[:, 0, :, :])
        loss_bb_regressors = mse_loss(yh_bboxes[b, 1:, i, j], y_bboxes[b, 1:, i, j])
        denom = 5
    
    return loss_n_bboxes, loss_bbox_spread, loss_segmentation, loss_bb_centroids, loss_bb_regressors, denom
    