#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python tools/test.py models/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024-snow.py models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth

# Create a loop to evaluate all datasets types
for dataset in rainy foggy spring autumn snow sunny dawn night 
do

    # Create a loop to evaluate all the models in the models folder
    for model in models/original/*.py
    do
        # Get the model name without .py extension
        model_name=$(basename $model .py)
        # Find checkpoint file which contains the model name
        checkpoint=$(find models -name "*$model_name*.pth")
        # Evaluate the model
        CUDA_VISIBLE_DEVICES=1 python tools/test.py $model $checkpoint --work-dir work_dirs/adverse_robustness_eval/${dataset}/$model_name \
        --cfg-options test_dataloader.dataset.data_root='data/' \
        --cfg-options test_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd-val/${dataset} \
        --cfg-options test_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val'
    done

done