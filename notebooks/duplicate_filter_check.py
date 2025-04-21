# import os

import os
import shutil
output_folder = "data/cityscapes-adverse-hd-filter/"

modification_types = set(
    filename.split("_")[0]
    for filename in os.listdir(output_folder)
    if filename.endswith("_accepted.txt") or filename.endswith("_rejected.txt")
)

def backup_and_clean_file(file_path, exclude_paths=set()):
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bckup"
        shutil.copy(file_path, backup_path)

        with open(file_path, 'r') as f:
            paths = f.read().splitlines()

        unique_paths = list(set(paths) - exclude_paths)

        with open(file_path, 'w') as f:
            f.write("\n".join(unique_paths))

        print(f"Cleaned {file_path}: {len(paths) - len(unique_paths)} duplicates removed.")

def check_and_clean_duplicates(modification_type):
    accepted_file = os.path.join(output_folder, f"{modification_type}_accepted.txt")
    rejected_file = os.path.join(output_folder, f"{modification_type}_rejected.txt")
    
    accepted_paths = []
    rejected_paths = []
    
    if os.path.exists(accepted_file):
        with open(accepted_file, 'r') as f:
            accepted_paths = f.read().splitlines()
    
    if os.path.exists(rejected_file):
        with open(rejected_file, 'r') as f:
            rejected_paths = f.read().splitlines()
    
    all_paths = accepted_paths + rejected_paths
    duplicates = set([path for path in all_paths if all_paths.count(path) > 1])
    cross_duplicates = set(accepted_paths) & set(rejected_paths)
    
    print(f"\n{modification_type}:")
    print(f"Total paths: {len(all_paths)} | Accepted: {len(accepted_paths)} - Rejected: {len(rejected_paths)}")
    if len(all_paths) > 0:
        print(f"Acceptance Rate: {(len(accepted_paths) / len(all_paths)) * 100:.2f}%")
    else:
        print("Acceptance Rate: N/A (No paths found)")
    print(f"Duplicate paths: {len(duplicates)}")
    print(f"Cross duplicates between accepted and rejected: {len(cross_duplicates)}")

    if duplicates or cross_duplicates:
        print("Duplicates found:")
        for path in duplicates:
            print(path)
        for path in cross_duplicates:
            print(path)
        backup_and_clean_file(accepted_file, cross_duplicates)
        backup_and_clean_file(rejected_file, cross_duplicates)
    else:
        print("No duplicates found")

for mod_type in modification_types:
    check_and_clean_duplicates(mod_type)
