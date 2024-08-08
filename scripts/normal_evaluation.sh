#!/bin/bash

# Create a loop to evaluate all the models in the models folder
for model in models/original/*.py
do
    # Get the model name without .py extension
    model_name=$(basename $model .py)
    # Find checkpoint file which contains the model name
    checkpoint=$(find models -name "*$model_name*.pth")
    # Evaluate the model
    CUDA_VISIBLE_DEVICES=2 python tools/test.py $model $checkpoint --work-dir work_dirs/normal/$model_name
done