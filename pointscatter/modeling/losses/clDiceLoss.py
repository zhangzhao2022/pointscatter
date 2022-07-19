import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weighted_loss


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))

    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


@weighted_loss
def binary_soft_clDice_loss(pred, label, smooth=1., iter_=5):
    assert pred.size() == label.size()
    num = pred.size()[0]
    pred_flat = pred.view(pred.size()[0], -1)
    label_flat = label.view(pred.size()[0], -1)
    pred_skel_flat = soft_skel(pred, iter_).view(pred.size()[0], -1)
    label_skel_flat = soft_skel(label, iter_).view(pred.size()[0], -1)
    tprec = ((label_flat * pred_skel_flat).sum(1) + smooth) / (pred_skel_flat.sum(1) + smooth)
    tsens = ((pred_flat * label_skel_flat).sum(1) + smooth) / (label_skel_flat.sum(1) + smooth)
    soft_cl = 2 * tprec * tsens / (tprec + tsens)
    return 1 - soft_cl.sum() / num


@weighted_loss
def clDice_loss(pred, target, smooth=1., class_weight=None, ignore_index=255, iter_=5):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(1, num_classes):
        if i != ignore_index:
            loss_clDice = binary_soft_clDice_loss(
                pred[:, i],
                target[..., i],
                smooth=smooth,
                iter_=iter_,
            )
            if class_weight is not None:
                loss_clDice *= class_weight[i]
            total_loss += loss_clDice
    return total_loss / (num_classes - 1)


@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(1, num_classes):
        if i != ignore_index:
            loss_dice = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                loss_dice *= class_weight[i]
            total_loss += loss_dice
    return total_loss / (num_classes - 1)


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
    return 1 - num / den


@LOSSES.register_module()
class clDiceLoss(nn.Module):
    def __init__(self,
                 alpha=0.,
                 smooth=1,
                 iter_=5,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_clDice',
                 **kwargs):
        super(clDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.iter_ = iter_
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        loss_dice = (1 - self.alpha) * self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        if self.alpha == 0.:
            return loss_dice
        else:
            loss_clDice = self.alpha * self.loss_weight * clDice_loss(
                pred,
                one_hot_target.float(),
                smooth=self.smooth,
                iter_=self.iter_,
                class_weight=class_weight,
                ignore_index=self.ignore_index
            )
            return loss_dice + loss_clDice

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
