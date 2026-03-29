import torch
import torch.nn as nn
import torch.nn.functional as F

# bce = nn.BCELoss(reduction='mean')  # for sigmoid output of FSP NET
bce = nn.BCEWithLogitsLoss(reduction='mean')  # for raw output of PFNET


def multi_bce(preds, gt):
    m_loss = bce(preds[3], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        loss += bce(preds[i], gt) * ((2 ** i) / 16)  # loss
        # loss += bce(preds[i], gt) * ((1+i) / 4)
    return loss + m_loss, m_loss


def single_bce(pred, gt):
    return bce(pred, gt)


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def multi_structure_loss(preds, gt):
    m_loss = structure_loss(preds[3], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        # loss += bce(preds[i], gt) * ((2 ** i) / 16)  # loss
        loss += (structure_loss(preds[i], gt)) * ((1+i) / 4)
    return loss + m_loss, m_loss


    """
 @Time    : 2021/7/6 14:31
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : loss.py
 @Function: Loss
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)
