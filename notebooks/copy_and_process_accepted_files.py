import os
import shutil
import argparse
from collections import defaultdict

def copy_files(file_list, base_output_dir):
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    with open(file_list, 'r') as f:
        paths = f.read().splitlines()
    
    for path in paths:
        if os.path.exists(path):
            filename = os.path.basename(path)
            city_name = filename.split('_')[0]
            dest_dir = os.path.join(base_output_dir, city_name)
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            shutil.copy(path, dest_dir)
            print(f"Copied {path} to {dest_dir}")
        else:
            print(f"File {path} does not exist")

def main(modification_type, filtered_path, output_dir):
    base_output_dir = os.path.join(output_dir, modification_type)
    
    accepted_file = f"{filtered_path}/{modification_type}_accepted.txt"
    copy_files(accepted_file, base_output_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy filtered data to output directory.")
    parser.add_argument('--type', type=str, required=True, help="Modification type (e.g., autumn, dawn).")
    parser.add_argument('--filtered_path', type=str, default="./data/cityscapes-adverse-hd-filter")
    parser.add_argument('--output_dir', type=str, help="Output directory.")
    
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.filtered_path
    
    if args.type == 'all':
        dataset = defaultdict(list)
        for filtered_data in os.listdir(args.filtered_path):
            full_data_path = os.path.join(args.filtered_path, filtered_data)
            if os.path.isdir(full_data_path) or not full_data_path.endswith('.txt'):
                continue
            filename = filtered_data.split('.')[0]
            type, accept_or_reject = filename.split('_')
            dataset[type].append(accept_or_reject)

        for dataset_type, accept_or_rejects in dataset.items():
            print(f"Processing {dataset_type}: {accept_or_rejects}")
            if 'accepted' in accept_or_rejects:
                main(dataset_type, args.filtered_path, output_dir)

    else:
        main(args.type, args.filtered_path, output_dir)
