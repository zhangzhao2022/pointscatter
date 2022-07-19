from configs.config import backbone_config, backbone_out_channels, clhead_alpha, cline_alpha, ps_feature_channels, ps_feature_index, M

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
        loss_decode=dict(type='clDiceLoss', alpha=clhead_alpha, iter_=10, loss_weight=1.0)),
    auxiliary_head=[dict(
        type='PointScatterHead',
        in_index=ps_feature_index,
        backbone_channels=ps_feature_channels,
        M=M,
        N=M ** 2,
        loss_decode=dict(type='PointLoss', loss_weight=1.0, alpha=cline_alpha, loss_name='loss_point_cline', ske_label=True))],
)
