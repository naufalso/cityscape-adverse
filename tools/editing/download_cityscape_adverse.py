import os
import argparse
from huggingface_hub import snapshot_download

def download_cityscape_adverse(save_dir):
    # Define the model repository and the local directory
    repo_id = "naufalso/cityscape-adverse"
    local_dir = save_dir

    # Download the training dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,

    )


def main():
    parser = argparse.ArgumentParser(description="Download Cityscapes Adverse dataset.")
    parser.add_argument(
        '--save-dir',
        type=str,
        default="./data/cityscapes-adverse/",
        help='Directory to save the dataset.'
    )
    args = parser.parse_args()

    print(f"Downloading Cityscapes Adverse dataset to {args.save_dir}...")
    download_cityscape_adverse(args.save_dir)
    print("Cityscapes Adverse dataset downloaded successfully.")

if __name__ == "__main__":
    main()