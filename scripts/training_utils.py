import numpy as np
import torch
import torch.nn as nn

from global_wheat_detection.scripts.preprocessing import DataLoader
import global_wheat_detection.scripts.utils as utils

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss(reduction='none')


def training_loss( yh_pretrained, yh_segmentation, yh_bboxes
                 , y_pretrained, y_segmentation, y_bboxes, bbox_class_wts
                 ):
    """ Combined training loss across training objectives
        (1): Match weights of pretained model
        (2): Regress on segmentation mask
        (3): Bounding box prediction
            (a): Classification of bounding box centroids
            (b): Centroid x positions regression
            (c): Centroid y positions regression
            (d): Bound box area ratios regression
            (e): Bounding box side ratios regression
    """

    loss_pretrained = mse_loss(yh_pretrained, y_pretrained)
    
    loss_segmentation = mse_loss(yh_segmentation, y_segmentation)

    loss_bb_classification = torch.sum(bce_loss(yh_bboxes[:,0,:,:], y_bboxes[:,0,:,:])*bbox_class_wts)
    
    loss = ( torch.clamp_max(loss_pretrained, 3) 
           + torch.clamp_max(loss_segmentation, 3) 
           + torch.clamp_max(loss_bb_classification,3)
           )

    b, i, j = torch.where(y_bboxes[:, 0, :, :] == 1)
    if len(i) > 0:
        loss_bb_regressors = mse_loss(yh_bboxes[b, 1:, i, j], y_bboxes[b, 1:, i, j])
        #TODO: Debug inf
        print(loss_pretrained.item(), loss_segmentation.item(), loss_bb_classification.item(), loss_bb_regressors.item())
        loss = loss + torch.clamp_max(loss_bb_regressors,3)
    
    return loss

def inference_output(yh_bboxes, w, h, threshold=0.5, max_boxes=200):
    yh_bboxes[:,0,:,:] = torch.sigmoid(yh_bboxes[:,0,:,:])
    yh_bboxes = yh_bboxes.numpy()
    b, *_ = yh_bboxes.shape
    
    bbs = []
    # [confidence, xmin, ymin, width, height]
    for im_idx in range(b):
        t = max(threshold, utils.large_n(yh_bboxes[im_idx, 0, :, :].flatten(), max_boxes))
        i, j = np.where(yh_bboxes[im_idx, 0, :, :] >= t)
        confidences = yh_bboxes[im_idx, 0, i, j]
        xs = yh_bboxes[im_idx, 1, i, j]
        ys = yh_bboxes[im_idx, 2, i, j]
        areas = yh_bboxes[im_idx, 3, i, j]
        sides = yh_bboxes[im_idx, 4, i, j]

        bbs.append(np.array([[c] + utils.bbox_pred_to_dims(x, y, a, s, w, h) 
                             for (c, x, y, a, s)
                             in zip(confidences, xs, ys, areas, sides)
                            ]))
            
    return bbs

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


    lr_sales = cyclic_lr_scales(n_epochs, n_warmup)
    def lr_lambda(epoch):
        """
        'We use a mini batch size of 64'
        https://arxiv.org/pdf/1704.00109.pdf
        """
        return lr_sales[epoch]/64
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)