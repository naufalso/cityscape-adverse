import os
import argparse
import numpy as np
from cityscapesscripts.helpers.labels import labels
from PIL import Image
from tqdm import tqdm

def convert_labelIds_to_trainIds(labelIds_image):
    """
    Converts a Cityscapes labelIds image to trainIds.

    Args:
        labelIds_image (numpy.ndarray): The input image where each pixel contains a labelId.

    Returns:
        numpy.ndarray: An output image where each pixel contains the corresponding trainId.
    """
    # Create a mapping from labelId to trainId
    labelId_to_trainId = {label.id: label.trainId for label in labels}

    # Create an array to store the trainIds
    trainIds_image = np.zeros_like(labelIds_image, dtype=np.uint8)

    # Apply the mapping from labelIds to trainIds
    for labelId, trainId in labelId_to_trainId.items():
        trainIds_image[labelIds_image == labelId] = trainId

    return trainIds_image

def process_folder(folder):
    """
    Converts all _labelIds images in the input folder and its subdirectories to _gtFine_labelTrainIds format
    and saves the results in the same folder.

    Args:
        folder (str): Path to the folder containing _labelIds images.
    """
    # Collect all _labelIds.png files in the folder and its subdirectories
    label_ids_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('_labelIds.png'):
                label_ids_files.append(os.path.join(root, file))

    # Process each file with a progress bar
    for label_ids_file in tqdm(label_ids_files, desc="Converting images"):
        output_path = label_ids_file.replace('_labelIds', '_gtFine_labelTrainIds')

        # Load the _labelIds image
        labelIds_image = np.array(Image.open(label_ids_file))

        # Convert to trainIds
        trainIds_image = convert_labelIds_to_trainIds(labelIds_image)

        # Save the result as a PNG image
        trainIds_pil = Image.fromarray(trainIds_image)
        trainIds_pil.save(output_path)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert Cityscapes _labelIds to _gtFine_labelTrainIds.")
    parser.add_argument('folder', type=str, help='Path to the folder containing _labelIds images and subdirectories.')

    # Parse the arguments
    args = parser.parse_args()

    # Process the folder
    process_folder(args.folder)

if __name__ == '__main__':
    main()
