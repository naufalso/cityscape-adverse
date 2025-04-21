import os

# Define the base directories
# base_dir = os.path.abspath('./data/cityscapes-adverse-hd-train')
# gt_base_dir = os.path.abspath('./data/cityscapes/gtFine/train')
# target_dir = os.path.join(base_dir, 'all')
# gt_target_dir = os.path.abspath('./data/cityscapes/gtFine/train-all')

base_dir = os.path.abspath('./data/cityscapes-adverse-hd')
gt_base_dir = os.path.abspath('./data/cityscapes/gtFine/val')
target_dir = os.path.join(base_dir, 'all')
gt_target_dir = os.path.abspath('./data/cityscapes/gtFine/val-all')

# Create the target directories if they don't exist
os.makedirs(target_dir, exist_ok=True)
os.makedirs(gt_target_dir, exist_ok=True)

# Function to create symbolic links for dataset images
def create_symlinks_for_images():
    for dataset_type in os.listdir(base_dir):
        dataset_type_path = os.path.join(base_dir, dataset_type)
        
        # Check if it is a directory
        if os.path.isdir(dataset_type_path) and dataset_type != 'all':
            # Iterate through the city names
            for city_name in os.listdir(dataset_type_path):
                city_path = os.path.join(dataset_type_path, city_name)
                
                # Check if it is a directory
                if os.path.isdir(city_path):
                    # Create the corresponding city directory in the target folder
                    target_city_path = os.path.join(target_dir, city_name)
                    os.makedirs(target_city_path, exist_ok=True)
                    
                    # Iterate through the dataset files
                    for dataset_file in os.listdir(city_path):
                        file_path = os.path.join(city_path, dataset_file)
                        
                        # Check if it is a file
                        if os.path.isfile(file_path):
                            # Split the original filename
                            name_parts = dataset_file.split('_')
                            
                            # Insert dataset_type before the last part
                            new_name_parts = name_parts[:-1] + [dataset_type] + [name_parts[-1]]
                            link_name = '_'.join(new_name_parts)
                            
                            link_path = os.path.join(target_city_path, link_name)
                            
                            # Create the symbolic link using absolute paths
                            os.symlink(file_path, link_path)

# Function to create symbolic links for ground truth files
def create_symlinks_for_ground_truth():
    for city_name in os.listdir(gt_base_dir):
        city_path = os.path.join(gt_base_dir, city_name)
        
        # Check if it is a directory
        if os.path.isdir(city_path):
            # Create the corresponding city directory in the target folder
            target_city_path = os.path.join(gt_target_dir, city_name)
            os.makedirs(target_city_path, exist_ok=True)
            
            # Iterate through the ground truth files
            for gt_file in os.listdir(city_path):
                gt_file_path = os.path.join(city_path, gt_file)
                
                # Check if it is a file and match the ground truth pattern
                if os.path.isfile(gt_file_path) and gt_file.endswith('_gtFine_labelTrainIds.png'):
                    # Get the base filename without the ground truth suffix
                    base_filename = gt_file.replace('_gtFine_labelTrainIds.png', '')
                    
                    # Create links for each dataset type
                    for dataset_type in os.listdir(base_dir):
                        if os.path.isdir(os.path.join(base_dir, dataset_type)) and dataset_type != 'all':
                            link_name = f"{base_filename}_{dataset_type}_gtFine_labelTrainIds.png"
                            link_path = os.path.join(target_city_path, link_name)
                            
                            # Create the symbolic link using absolute paths
                            os.symlink(gt_file_path, link_path)

# Create symbolic links for both images and ground truth
create_symlinks_for_images()
create_symlinks_for_ground_truth()

print("Symbolic links created successfully.")
