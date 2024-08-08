#!/bin/bash

model_path=/home/islab-ai/naufal/mmsegmentation/models/augmentation_evaluation

# Create a loop to train all config types
for config in $model_path/*.py
do
    # Get the config name without .py extension
    config_name=$(basename $config .py)
    # Train the model
    # CUDA_VISIBLE_DEVICES=3 python tools/train.py models/augmentation_evaluation/deeplabv3plus_gaussianblur.py --work-dir work_dirs/augmentation_train/gaussian_blur
    CUDA_VISIBLE_DEVICES=3 python tools/train.py $config --work-dir work_dirs/augmentation_train/$config_name
done