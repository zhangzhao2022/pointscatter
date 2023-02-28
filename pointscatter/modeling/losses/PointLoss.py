import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from fvcore.nn import sigmoid_focal_loss_jit

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weighted_loss
from pointscatter.modeling.layers.coords import get_centered_meshgrid, batched_greedy_assignment, Hungarian_assignment


@LOSSES.register_module()
class PointLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=0.6, loss_name='loss_point', ske_label=False, **kwargs):
        super(PointLoss, self).__init__()
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.ske_label = ske_label

    def forward(self, pred, target):
        pred_scores, pred_offsets, H, W, M = pred
        B, _, _ = target.shape

        target = target.gt(0).long()

        if self.ske_label:
            target = target.cpu().numpy()
            target = np.stack([skeletonize(target[t]).astype(np.uint8) for t in range(target.shape[0])])
            target = torch.from_numpy(target).to(pred_scores.device).long()

        target = target.view(B, H, M, W, M)

        mask_offsets = get_centered_meshgrid((M, M), pred_scores.device)
        mask_offsets = mask_offsets.view(-1, 2) / torch.tensor(M, device=mask_offsets.device)

        block_target = target.permute(0, 1, 3, 2, 4).contiguous().reshape(B, H * W, -1)
        block_target, indices = torch.sort(block_target, dim=-1, descending=True, stable=True)
        mask_offsets = mask_offsets[indices]

        loc_cost = (pred_offsets[..., None, :] - mask_offsets[..., None, :, :]).abs().sum(-1)
        cls_cost = (pred_scores[..., None].sigmoid() - block_target[..., None, :].float()).abs()
        cost = loc_cost.pow(0.8) * cls_cost.pow(0.2)
        cost = cost * block_target[..., None, :]

        # compute cost
        assignment = batched_greedy_assignment(cost)
        # assignment = Hungarian_assignment(cost)

        target_scores = torch.zeros_like(pred_scores, dtype=block_target.dtype)
        target_scores.scatter_(dim=-1, index=assignment, src=block_target)
        target_offsets = torch.zeros_like(pred_offsets)
        target_offsets.scatter_(dim=-2, index=torch.stack(2 * [assignment], -1), src=mask_offsets)

        # Update normalizer
        num_pos = max(target_scores.count_nonzero(), 1)
        normalizer = self._ema_update("loss_normalizer", num_pos, num_pos)

        obj_loss = sigmoid_focal_loss_jit(
            pred_scores,
            target_scores.float(),
            alpha=self.alpha,
            gamma=2.0,
            reduction="sum",
        )

        loc_loss = (pred_offsets - target_offsets).abs()
        loc_loss = (loc_loss.sum(-1) * target_scores.float()).sum()

        return self.loss_weight * (obj_loss / normalizer + 10 * loc_loss / normalizer)

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

    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new
