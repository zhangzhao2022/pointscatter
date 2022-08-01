# PointScatter based on MMSegmentation

An alternative of the segmentation models for the tubular structure extraction task.


# Installation


1.First install MMSegmentation following [mmseg_github](https://github.com/open-mmlab/mmsegmentation).

2.To install `pointscatter`, run `python setup.py develop`


# Dataset Preparation
The location of the datasets should be the same as the path indicated in the corresponding file in `"./configs/_base_/datasets/"`.

For DRIVE and STARE, refer to `mmsegmentation` for preparation process.

Massachusetts Roads can be downloaded from [massroad_kaggle](https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset). We select a subset of 694 images to train, which is indicated in `"./configs/_base_/datasets/massroads_split/"`, and test the model on the 13 official test images.

DeepGlobe can be downloaded from [deepglobe_challenge](https://competitions.codalab.org/competitions/18467). We split it by `"./configs/_base_/datasets/deepglobe_split/"`.

You may need to prepare the two road datasets according to the following structure after downloading them.
```
mass_roads_choose
│
└───training
│   └───input
│   └───output
│   └───output_cline
└───testing
│   └───...
```


# Train and Test

Train `pointscatter` with a single GPU:
```
python train.py configs/segmentors/3_pointscatter.py --dataset drive --backbone unet --work-dir ../output/drive_log/unet_3_pointscatter
```

Train with multiple GPUs:
```
python -m torch.distributed.run --nproc_per_node=4 --master_port=8888 train.py configs/segmentors/3_pointscatter.py --dataset drive --backbone unet --work-dir ../output/drive_log/unet_3_pointscatter --launcher pytorch
```

Test a trained model:
```
python test.py configs/segmentors/3_pointscatter.py  ../output/drive_log/unet_3_pointscatter/latest.pth --dataset drive --backbone unet --work-dir ../output/drive_log/unet_3_pointscatter --eval mDice
```

# Customize models
You can customize your own model by:
```
1. Add a new backbone in "./pointscatter/modeling/backbones" or register a new dataset in "./pointscatter/datasets".
2. Configure the new model components in "./configs/".
3. Train the model.
```
