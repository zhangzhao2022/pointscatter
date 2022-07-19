from configs.info import dataset_name, backbone_name

# configs defined by backbone
backbone_configs = {
    'unet': dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='DeconvModule'),
        norm_eval=False),
    'unet_pp': dict(
        type='UNet_PP',
        in_channels=3,
        base_channels=64,
        act_cfg=dict(type='ReLU'),
        norm_eval=False),
    'res_unet': dict(
        type='ResUNet',
        act_cfg=dict(type='ReLU')),
    'linknet34': dict(
        type='LinkNet34',
        act_cfg=dict(type='ReLU')),
    'dinknet34': dict(
        type='DinkNet34',
        act_cfg=dict(type='ReLU')),
}

backbone_anchor_channels = {
    'unet': (64, 256, -3, 4),
    'unet_pp': (64, 256, -3, 4),
    'res_unet': (64, 256, -3, 4),
    'linknet34': (32, 128, -4, 8),
    'dinknet34': (32, 128, -4, 8),
}

backbone_config = backbone_configs[backbone_name]
backbone_out_channels, ps_feature_channels, ps_feature_index, M = backbone_anchor_channels[backbone_name]

# configs defined by dataset
test_stride = {
    'drive': ((384, 384), (256, 256)),
    'stare': ((384, 384), (320, 320)),
    'massroads': ((1024, 1024), (500, 500)),
    'deepglobe': ((768, 768), (256, 256)),
}

lr_schedules = {
    'drive': '../_base_/schedules/schedule_3k.py',
    'stare': '../_base_/schedules/schedule_3k.py',
    'massroads': '../_base_/schedules/schedule_10k.py',
    'deepglobe': '../_base_/schedules/schedule_40k.py',
}

pshead_alphas = {
    'drive': (0.6, 0.7),
    'stare': (0.6, 0.7),
    'massroads': (0.6, 0.8),
    'deepglobe': (0.6, 0.8),
}

clhead_alphas = {
    'drive': 0.3,
    'stare': 0.3,
    'massroads': 0.3,
    'deepglobe': 0.1,
}

crop_size_test, stride_test = test_stride[dataset_name]
lr_schedule = lr_schedules[dataset_name]
seg_alpha, cline_alpha = pshead_alphas[dataset_name]
clhead_alpha = clhead_alphas[dataset_name]

# other configs
aux_loss_weight = 1.0
