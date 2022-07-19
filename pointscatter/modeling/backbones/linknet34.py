import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES
from mmcv.cnn import build_norm_layer, build_activation_layer, kaiming_init


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, act_cfg=dict(type='ReLU')):
        super(DecoderBlock, self).__init__()
        self.activation = build_activation_layer(act_cfg)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = self.activation

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = self.activation

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = self.activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


@BACKBONES.register_module()
class LinkNet34(BaseModule):
    def __init__(self,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 pretrained=None,
                 ):
        super(LinkNet34, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.pretrained = pretrained
        self.activation = build_activation_layer(act_cfg)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1, output_padding=1)
        self.finalrelu1 = self.activation
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = self.activation

    def forward(self, x):
        self._check_input_divisible(x)
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # 4*, 64
        e1 = self.encoder1(x)   # 4*, 64
        e2 = self.encoder2(e1)  # 8*, 128
        e3 = self.encoder3(e2)  # 16*, 256
        e4 = self.encoder4(e3)  # 32*, 512

        # Decoder
        d4 = self.decoder4(e4)  # 16*, 256
        d3 = self.decoder3(d4 + e3)  # 8*, 128
        d2 = self.decoder2(d3 + e2)   # 4*, 64
        d1 = self.decoder1(d2 + e1)  # 2*, 64
        out = self.finaldeconv1(d1)  # 1*, 32
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)

        return [e4, d4, d3, d2, d1, out]

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(LinkNet34, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 8
        assert (h % whole_downsample_rate == 0) \
               and (w % whole_downsample_rate == 0), \
            f'The input image size {(h, w)} should be divisible by the whole ' \
            f'downsample rate {whole_downsample_rate}.'
