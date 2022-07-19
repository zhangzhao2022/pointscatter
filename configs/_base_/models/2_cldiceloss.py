from configs.config import backbone_config, backbone_out_channels, clhead_alpha

# model settings
find_unused_parameters = True
norm_cfg = dict(type='BN', requires_grad=True)
backbone_config.update(dict(norm_cfg=norm_cfg))
model = dict(
    type='EncoderDecoderVisual',
    pretrained=None,
    backbone=backbone_config,
    decode_head=dict(
        type='FCNHead',
        in_channels=backbone_out_channels,
        in_index=-1,
        channels=backbone_out_channels,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='clDiceLoss', alpha=clhead_alpha, iter_=10, loss_weight=1.0))
)
