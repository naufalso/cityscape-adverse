#!/bin/bash
cross_evaluation_dir=work_dirs/cross_evaluation

# Define the datasets array
trained_models=("original" "spring" "autumn" "snow" "rainy" "night" "foggy" "dawn" "sunny" "all")
eval_datasets=("original" "spring" "autumn" "snow" "rainy" "night" "foggy" "dawn" "sunny" "all")

# Total number of iterations
total_iterations=$((${#trained_models[@]} * ${#eval_datasets[@]}))

# Initialize the counter
counter=1

# Create a loop to evaluate all datasets types
for trained_model in "${trained_models[@]}"
do
    # Create a loop to evaluate all the models in the models folder
    for eval_dataset in "${eval_datasets[@]}"
    do
        # Get the model name without .py extension
        model_name=deeplabv3plus_${trained_model}

        # Get the train model config path
        model_path=${cross_evaluation_dir}/${model_name}/${model_name}.py

        # Last checkpoint file
        last_checkpoint_file=${cross_evaluation_dir}/${model_name}/last_checkpoint

        # Find the last checkpoints
        last_checkpoint_path=$(<"$last_checkpoint_file")

        # Use the if statement to compare the variable
        if [ "$eval_dataset" == "original" ]; then
            test_data_prefix="cityscapes/leftImg8bit/val"
        else
            test_data_prefix="cityscapes-adverse-hd/${eval_dataset}"
        fi

        if [ "$eval_dataset" == "all" ]; then
            seg_map_path="cityscapes/gtFine/val-all"
        else
            seg_map_path="cityscapes/gtFine/val"
        fi

        echo "(${counter}/${total_iterations}) Evaluating ${model_name} on ${eval_dataset} datasets"

        CUDA_VISIBLE_DEVICES=2 python tools/test.py $model_path $last_checkpoint_path \
        --work-dir work_dirs/filtered_cross_evaluation_eval/deeplabv3plus/deeplabv3plus_${trained_model}_${eval_dataset} \
        --cfg-options test_dataloader.dataset.data_root='data/' \
        --cfg-options test_dataloader.dataset.data_prefix.img_path=$test_data_prefix \
        --cfg-options test_dataloader.dataset.data_prefix.seg_map_path=$seg_map_path
        # --cfg-options visualizer.vis_backends.1.init_kwargs.name=filtered_eval_${trained_model}_${eval_dataset} \
        # --cfg-options visualizer.vis_backends.1.init_kwargs.tags=[deeplabv3,cnn,r-50,${trained_model},cross-evaluation,filtered_eval_${eval_dataset},evaluation]

        # Increment the counter
        counter=$((counter + 1))
    done
done



# #!/bin/bash

# # CUDA_VISIBLE_DEVICES=1 python tools/test.py models/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024-snow.py models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth

# cross_evaluation_dir=work_dirs/cross_evaluation

# # Create a loop to evaluate all datasets types
# for trained_model in original spring autumn snow rainy night foggy dawn sunny
# do
#     # Create a loop to evaluate all the models in the models folder
#     for eval_dataset in original spring autumn snow rainy night foggy dawn sunny
#     do
#         # Get the model name without .py extension
#         model_name=deeplabv3plus_${trained_model}

#         # Get the train model config path
#         model_path=${cross_evaluation_dir}/${model_name}/${model_name}.py

#         # Last checkpoint file
#         last_checkpoint_file=${cross_evaluation_dir}/${model_name}/last_checkpoint

#         # Find the last checkpoints
#         last_checkpoint_path=$(<"$last_checkpoint_file")

#         # Use the if statement to compare the variable
#         if [ "$eval_dataset" == "original" ]; then
#             test_data_prefix=cityscapes/leftImg8bit/train
#         else
#             test_data_prefix=cityscapes-adverse-hd/${eval_dataset}
#         fi

#         echo Evaluating ${model_name} on ${eval_dataset} datasets

#         CUDA_VISIBLE_DEVICES=3 python tools/test.py $model_path $last_checkpoint_path \
#         --work-dir work_dirs/cross_evaluation_eval/deeplabv3plus_${trained_model}_${eval_dataset} \
#         --cfg-options test_dataloader.dataset.data_root='data/' \
#         --cfg-options $test_data_prefix \
#         --cfg-options test_dataloader.dataset.data_prefix.seg_map_path='cityscapes/gtFine/val' \
#         --cfg-options visualizer.vis_backends.1.init_kwargs.name=eval_${model_name}_${trained_model}_${eval_dataset} \
#         --cfg-options visualizer.vis_backends.1.init_kwargs.tags=[segformer,transformer,mit-b0,${dataset},cross-evaluation,eval_${eval_dataset},evaluation]
#     done

# done