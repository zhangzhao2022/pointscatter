from configs.config import dataset_name, lr_schedule, crop_size_test, stride_test

model_file = '../_base_/models/7_cline_pointscatter.py'
dataset_file = '../_base_/datasets/' + dataset_name + '_cline.py'
run_file = '../_base_/default_runtime.py'

_base_ = [model_file, dataset_file, run_file, lr_schedule]
model = dict(test_cfg=dict(mode='slide', crop_size=crop_size_test, stride=stride_test))
evaluation = dict(metric=['mDice'])
