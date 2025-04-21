#!/bin/bash
# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

# Define the maximum of parallel process in the queue
N=2
open_sem $N 

model_path=/home/islab-ai/naufal/mmsegmentation/models/cross_evaluation

export CUDA_VISIBLE_DEVICES=3

# Create a loop to train all config types
for config_name in deeplabv3plus_all # deeplabv3plus_spring deeplabv3plus_snow deeplabv3plus_rainy deeplabv3plus_foggy deeplabv3plus_night deeplabv3plus_dawn deeplabv3plus_sunny
do
    config=${model_path}/${config_name}.py
    # Train the model
    run_with_lock python tools/train.py $config --work-dir work_dirs/cross_evaluation/$config_name
done

# Original dataset training
# run_with_lock python tools/train.py models/deeplabv3plus_original.py --work-dir work_dirs/cross_evaluation/deeplabv3plus_original

# # Create a loop to train all config types
# for config in $model_path/*.py
# do
#     # Get the config name without .py extension
#     config_name=$(basename $config .py)
#     # Train the model
#     run_with_lock python tools/train.py $config --work-dir work_dirs/cross_evaluation/$config_name
# done