import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models.builder import BACKBONES
from mmcv.cnn import build_norm_layer, build_activation_layer, kaiming_init, constant_init


class DoubleConvBlock(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(DoubleConvBlock, self).__init__()
        self.activation = build_activation_layer(act_cfg)
        self.act_cfg = act_cfg
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.norm_name, self.norm1 = build_norm_layer(norm_cfg, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        _, self.norm2 = build_norm_layer(norm_cfg, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        output = self.activation(x)

        return output


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(DeconvBlock, self).__init__()
        self.activation = build_activation_layer(act_cfg)
        self.norm_name, self.norm = build_norm_layer(norm_cfg, out_ch)
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        output = self.activation(x)

        return output


@BACKBONES.register_module()
class UNet_PP(BaseModule):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 pretrained=None,
                 init_cfg=None):
        super(UNet_PP, self).__init__()

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

        n1 = base_channels
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up11 = DeconvBlock(filters[4], filters[3], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up21 = DeconvBlock(filters[3], filters[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up22 = DeconvBlock(filters[3], filters[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up31 = DeconvBlock(filters[2], filters[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up32 = DeconvBlock(filters[2], filters[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up33 = DeconvBlock(filters[2], filters[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up41 = DeconvBlock(filters[1], filters[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up42 = DeconvBlock(filters[1], filters[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up43 = DeconvBlock(filters[1], filters[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Up44 = DeconvBlock(filters[1], filters[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.Up = nn.Upsample(scale_factor=2.0, mode=upsample_mode, align_corners=True)

        factor = 2

        self.conv0_0 = self.conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = self.conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = self.conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = self.conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = self.conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = self.conv_block_nested(filters[0] + filters[1] // factor, filters[0], filters[0])
        self.conv1_1 = self.conv_block_nested(filters[1] + filters[2] // factor, filters[1], filters[1])
        self.conv2_1 = self.conv_block_nested(filters[2] + filters[3] // factor, filters[2], filters[2])
        self.conv3_1 = self.conv_block_nested(filters[3] + filters[4] // factor, filters[3], filters[3])

        self.conv0_2 = self.conv_block_nested(filters[0] * 2 + filters[1] // factor, filters[0], filters[0])
        self.conv1_2 = self.conv_block_nested(filters[1] * 2 + filters[2] // factor, filters[1], filters[1])
        self.conv2_2 = self.conv_block_nested(filters[2] * 2 + filters[3] // factor, filters[2], filters[2])

        self.conv0_3 = self.conv_block_nested(filters[0] * 3 + filters[1] // factor, filters[0], filters[0])
        self.conv1_3 = self.conv_block_nested(filters[1] * 3 + filters[2] // factor, filters[1], filters[1])

        self.conv0_4 = self.conv_block_nested(filters[0] * 4 + filters[1] // factor, filters[0], filters[0])

    def conv_block_nested(self, in_ch, mid_ch, out_ch):
        return DoubleConvBlock(in_ch, mid_ch, out_ch, self.norm_cfg, self.act_cfg)

    def forward(self, x):
        self._check_input_divisible(x)
        x0_0 = self.conv0_0(x)  # 1, filters[0]
        x1_0 = self.conv1_0(self.pool(x0_0))  # 2, filters[1]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up41(x1_0)], 1))  # 1, , filters[0]

        x2_0 = self.conv2_0(self.pool(x1_0))  # 4, , filters[2]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up31(x2_0)], 1))  # 2, filters[1]
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up42(x1_1)], 1))  # 1, filters[0]

        x3_0 = self.conv3_0(self.pool(x2_0))  # 8, filters[3]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up21(x3_0)], 1))  # 4, filters[2]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up32(x2_1)], 1))  # 2, filters[1]
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up43(x1_2)], 1))  # 1, filters[0]

        x4_0 = self.conv4_0(self.pool(x3_0))  # 16, filters[4]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up11(x4_0)], 1))  # 8, filters[3]
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up22(x3_1)], 1))  # 4, filters[2]
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up33(x2_2)], 1))  # 2, filters[1]
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up44(x1_3)], 1))  # 1, filters[0]

        output = [x4_0, x3_1, x2_2, x1_3, x0_4]

        return output

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet_PP, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 16
        assert (h % whole_downsample_rate == 0) \
               and (w % whole_downsample_rate == 0), \
            f'The input image size {(h, w)} should be divisible by the whole ' \
            f'downsample rate {whole_downsample_rate}.'
