from configs.config import backbone_config, seg_alpha, ps_feature_channels, ps_feature_index, M

# model settings
find_unused_parameters = True
norm_cfg = dict(type='BN', requires_grad=True)
backbone_config.update(dict(norm_cfg=norm_cfg))
model = dict(
    type='EncoderDecoderVisual',
    pretrained=None,
    backbone=backbone_config,
    decode_head=dict(
        type='PointScatterHead',
        in_index=ps_feature_index,
        backbone_channels=ps_feature_channels,
        M=M,
        N=M ** 2,
        loss_decode=dict(type='PointLoss', loss_weight=1.0, alpha=seg_alpha, loss_name='loss_point_seg'))
)
