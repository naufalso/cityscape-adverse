import os
import shutil
import argparse
from collections import defaultdict

def copy_files(file_list, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    with open(file_list, 'r') as f:
        paths = f.read().splitlines()
    
    for path in paths:
        if os.path.exists(path):
            shutil.copy(path, dest_dir)
            print(f"Copied {path} to {dest_dir}")
        else:
            print(f"File {path} does not exist")

def main(modification_type, filtered_path, is_accepted, is_rejected, output_dir):
    base_output_dir = os.path.join(output_dir, modification_type)
    
    if is_accepted:
        accepted_file = f"{filtered_path}/{modification_type}_accepted.txt"
        accepted_output_dir = os.path.join(base_output_dir, 'accepted')
        copy_files(accepted_file, accepted_output_dir)
    
    if is_rejected:
        rejected_file = f"{filtered_path}/{modification_type}_rejected.txt"
        rejected_output_dir = os.path.join(base_output_dir, 'rejected')
        copy_files(rejected_file, rejected_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy filtered data to output directory.")
    parser.add_argument('--type', type=str, required=True, help="Modification type (e.g., autumn, dawn).")
    parser.add_argument('--filtered_path', type=str, default="./data/cityscapes-adverse-hd-filter")
    parser.add_argument('--is_accepted', action='store_true', help="Flag to copy accepted files.")
    parser.add_argument('--is_rejected', action='store_true', help="Flag to copy rejected files.")
    parser.add_argument('--output_dir', type=str, help="Output directory.")
    # parser.add_argument('--all', action='store_true', help="Flag to copy rejected files.")
    
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
            main(dataset_type, args.filtered_path, 'accepted' in accept_or_rejects, 'rejected' in accept_or_rejects, output_dir)

    else:
        main(args.type, args.filtered_path, args.is_accepted, args.is_rejected, args.filtered_path, output_dir)
