import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.upsample(x)
        return out



@BACKBONES.register_module()
class ResUNet(BaseModule):
    def __init__(self,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_mode='deconv',
                 norm_eval=False,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(ResUNet, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        filters = [64, 128, 256, 512]
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        factor = 1

        self.upsample_1 = Upsample(filters[3], filters[3] // factor)
        self.up_residual_conv1 = ResidualConv(filters[3]//factor + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2] // factor)
        self.up_residual_conv2 = ResidualConv(filters[2] // factor + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1] // factor)
        self.up_residual_conv3 = ResidualConv(filters[1] // factor + filters[0], filters[0], 1, 1)

    def forward(self, x):
        self._check_input_divisible(x)
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)  # 1*, 64
        x2 = self.residual_conv_1(x1)  # 2*, 128
        x3 = self.residual_conv_2(x2)  # 4*, 256
        # Bridge
        x4 = self.bridge(x3)  # 8*, 512
        # Decode
        x5 = self.upsample_1(x4)
        x5 = torch.cat([x5, x3], dim=1)

        x6 = self.up_residual_conv1(x5)  # 4*, 256

        x7 = self.upsample_2(x6)
        x7 = torch.cat([x7, x2], dim=1)

        x8 = self.up_residual_conv2(x7)  # 2*, 128

        x9 = self.upsample_3(x8)
        x9 = torch.cat([x9, x1], dim=1)

        x10 = self.up_residual_conv3(x9)  # 1*, 64

        return [x4, x6, x8, x10]

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResUNet, self).train(mode)
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
