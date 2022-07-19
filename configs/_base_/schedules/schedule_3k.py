# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=250,
                 warmup_ratio=0.1,
                 min_lr_ratio=1e-5)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3000)
checkpoint_config = dict(by_epoch=False, interval=3000)
evaluation = dict(interval=3000, metric='mDice', pre_eval=True)
