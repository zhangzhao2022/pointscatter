import torch
import torch.nn as nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from pointscatter.modeling.layers.mlp import MLP


@HEADS.register_module()
class PointScatterHead(BaseDecodeHead):

    def __init__(self,
                 backbone_channels: int = 256,
                 N: int = 16,
                 M: int = 4,
                 **kwargs):
        super(PointScatterHead, self).__init__(in_channels=64, channels=64, num_classes=2, **kwargs)
        self.M = M
        self.N = N
        self.obj_head = nn.Linear(backbone_channels, self.N)
        self.loc_head = MLP(backbone_channels, backbone_channels, self.N * 2, 3)

    def forward(self, inputs):
        """Forward function."""
        features = self._transform_inputs(inputs)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).contiguous()
        L = H * W

        features = features.view(B, L, C)
        scores = self.obj_head(features)
        offsets = self.loc_head(features).view(B, L, self.N, 2)
        output = (scores, offsets, H, W, self.M)
        return output

    def losses(self, ps_head_pred, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(ps_head_pred, seg_label)
            else:
                loss[loss_decode.loss_name] += loss_decode(ps_head_pred, seg_label)
        return loss
