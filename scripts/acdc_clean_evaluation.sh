#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python tools/test.py models/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024-snow.py models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth

# Create a loop to evaluate all datasets types
for dataset in fog night rain snow
do

    # Create a loop to evaluate all the models in the models folder
    for model in models/original/*.py
    do
        # Get the model name without .py extension
        model_name=$(basename $model .py)
        # Find checkpoint file which contains the model name
        checkpoint=$(find models -name "*$model_name*.pth")
        # Evaluate the model
        CUDA_VISIBLE_DEVICES=2 python tools/test.py $model $checkpoint --work-dir work_dirs/acdc_clean/${dataset}/$model_name \
        --cfg-options test_dataloader.dataset.type='ACDCDataset' \
        --cfg-options test_dataloader.dataset.data_root=data/acdc_processed_clean/${dataset}/ \
        --cfg-options test_dataloader.dataset.data_prefix.img_path=rgb_anno/test \
        --cfg-options test_dataloader.dataset.data_prefix.seg_map_path=gt/test
    done

done