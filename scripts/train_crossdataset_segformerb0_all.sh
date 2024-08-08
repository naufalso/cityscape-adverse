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
N=1
open_sem $N 



export CUDA_VISIBLE_DEVICES=2

config=models/segformer_mit-b0_original.py
model_name=segformer_mit-b0

# # Original dataset training
# run_with_lock python tools/train.py $config --work-dir work_dirs/cross_evaluation_segformer/${model_name}_original \
# --resume \
# --cfg-options train_dataloader.batch_size=4 \
# --cfg-options visualizer.vis_backends.1.init_kwargs.name=${model_name}_original \
# --cfg-options visualizer.vis_backends.1.init_kwargs.tags=[segformer,transformer,mit-b0,original,cross-evaluation] \


# for dataset in spring autumn snow rainy night foggy sunny dawn  
# do
#     # Train the model
#     run_with_lock python tools/train.py $config --work-dir work_dirs/cross_evaluation_segformer/${model_name}_${dataset} \
#     --resume \
#     --cfg-options train_dataloader.batch_size=4 \
#     --cfg-options train_dataloader.dataset.data_root='data/' \
#     --cfg-options train_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd-train/${dataset} \
#     --cfg-options train_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/train' \
#     --cfg-options val_dataloader.dataset.data_root='data/' \
#     --cfg-options val_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd/${dataset} \
#     --cfg-options val_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val' \
#     --cfg-options test_dataloader.dataset.data_root='data/' \
#     --cfg-options test_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd/${dataset} \
#     --cfg-options test_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val' \
#     --cfg-options visualizer.vis_backends.1.init_kwargs.name=${model_name}_${dataset} \
#     --cfg-options visualizer.vis_backends.1.init_kwargs.tags=[segformer,transformer,mit-b0,${dataset},cross-evaluation]
# done


# Train the model
dataset=all
run_with_lock python tools/train.py $config --work-dir work_dirs/cross_evaluation_segformer/${model_name}_${dataset} \
--resume \
--cfg-options train_dataloader.batch_size=4 \
--cfg-options train_dataloader.dataset.data_root='data/' \
--cfg-options train_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd-train/${dataset} \
--cfg-options train_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/train-all' \
--cfg-options val_dataloader.dataset.data_root='data/' \
--cfg-options val_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd/${dataset} \
--cfg-options val_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val-all' \
--cfg-options test_dataloader.dataset.data_root='data/' \
--cfg-options test_dataloader.dataset.data_prefix.img_path=cityscapes-adverse-hd/${dataset} \
--cfg-options test_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val-all' \
--cfg-options visualizer.vis_backends.1.init_kwargs.name=${model_name}_${dataset} \
--cfg-options visualizer.vis_backends.1.init_kwargs.tags=[segformer,transformer,mit-b0,${dataset},cross-evaluation]