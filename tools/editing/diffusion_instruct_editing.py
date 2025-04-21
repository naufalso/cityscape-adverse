import os
import argparse
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Image as ImageDataset
from diffusers import EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from diffusers.utils import logging

# Disable progress bar logging for cleaner output
logging.disable_progress_bar()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Edit images using CosXL InstructPix2Pix model.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/cityscapes/leftImg8bit/val",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/cityscapes-adverse-hd-script",
        help="Directory to save edited images.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./models/huggingface",
        help="Directory for Hugging Face model cache.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use. Set to 'cpu' to force CPU usage.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN", ""), # Load from env var HF_TOKEN by default
        help="Hugging Face API token. Defaults to the value of the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Guidance scale for the diffusion model.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2048,
        help="Resolution to resize images to before processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, # Adjust based on GPU memory
        help="Batch size for processing images with datasets.map.",
    )
    return parser.parse_args()

def load_model(cache_dir, hf_token, device):
    """Loads the CosXL edit model and VAE."""
    os.environ["HF_TOKEN"] = hf_token
    edit_file = hf_hub_download(
        repo_id="stabilityai/cosxl",
        filename="cosxl_edit.safetensors",
        cache_dir=cache_dir,
    )
    # Determine dtype based on device
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        edit_file,
        num_in_channels=8,
        is_cosxl_edit=True,
        vae=vae,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    pipe_edit.scheduler = EDMEulerScheduler(
        sigma_min=0.002,
        sigma_max=120.0,
        sigma_data=1.0,
        prediction_type="v_prediction",
        sigma_schedule="exponential",
    )
    pipe_edit.to(device)
    pipe_edit.set_progress_bar_config(disable=True) # Disable tqdm for pipeline
    return pipe_edit

def resize_image(image, resolution):
    """Resizes an image while maintaining aspect ratio."""
    original_width, original_height = image.size
    max_dim = max(original_width, original_height)
    scale_factor = resolution / max_dim
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Ensure dimensions are multiples of 8 for compatibility with some models/pipelines
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    if new_width == 0 or new_height == 0:
         # Handle cases where scaling results in zero dimensions
         # Fallback to a minimum size or keep original if very small
         min_size = 8
         if original_width > original_height:
             new_width = max(min_size, (resolution // 8) * 8)
             aspect_ratio = original_height / original_width
             new_height = max(min_size, int(new_width * aspect_ratio) // 8 * 8)
         else:
             new_height = max(min_size, (resolution // 8) * 8)
             aspect_ratio = original_width / original_height
             new_width = max(min_size, int(new_height * aspect_ratio) // 8 * 8)


    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

def run_edit(pipe, image, prompt, negative_prompt="", guidance_scale=7, steps=20, resolution=2048):
    """Runs the image editing pipeline on a single image."""
    image = resize_image(image, resolution)
    width, height = image.size
    # Ensure width and height are not zero after resizing potentially small images
    if width == 0 or height == 0:
        print(f"Skipping image due to zero dimension after resize: Original size {image.size}")
        return None # Or handle appropriately

    return pipe(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images[0]

def main():
    args = parse_args()

    # Determine device
    if args.gpu_id.lower() == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = "cuda"
    else:
        print("Warning: CUDA not available, falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")


    # --- Configuration ---
    environment_variation_prompts = [
        "change the weather to rainy",
        "change the weather to foggy",
        "change the season to spring",
        "change the season to autumn",
        "change the season to winter with snow",
        "change the weather to bright sunny day",
        "change to night scene with lights on",
        "change to dawn scene",
        # Add more prompts as needed
    ]

    environment_variation_names = [
        "rainy",
        "foggy",
        "spring",
        "autumn",
        "snow",
        "sunny",
        "night",
        "dawn",
        # Corresponding names for saving directories
    ]

    if len(environment_variation_prompts) != len(environment_variation_names):
        raise ValueError("Number of prompts must match number of variation names.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    # --- Load Model ---
    print("Loading model...")
    pipe_edit = load_model(str(cache_dir), args.hf_token, device)
    print("Model loaded.")

    # --- Prepare Data ---
    print("Scanning for input images...")
    validation_images_path = []
    input_cities = [d for d in input_dir.iterdir() if d.is_dir()]
    if not input_cities: # Handle case where images are directly in input_dir
        input_cities = [input_dir]

    for city_dir in input_cities:
        city_name = city_dir.name
        # Create output directories per variation and city
        for env_var_name in environment_variation_names:
            (output_dir / env_var_name / city_name).mkdir(parents=True, exist_ok=True)

        for image_path in city_dir.glob("*.png"): # Assuming png format
             validation_images_path.append({"image": str(image_path)})


    if not validation_images_path:
        print(f"Error: No images found in {input_dir} or its subdirectories.")
        return

    print(f"Found {len(validation_images_path)} images.")

    # --- Create Dataset ---
    validation_dataset = Dataset.from_list(validation_images_path)
    # Cast to Image type for automatic loading by datasets
    validation_dataset = validation_dataset.cast_column("image", ImageDataset())
    print("Dataset created.")

    # --- Define Processing Function ---
    def edit_environment_variation(sample):
        """Applies all environment variations to a single image sample."""
        try:
            image = sample["image"]
            if image is None:
                print(f"Warning: Skipping sample, failed to load image: {sample.get('image_path', 'Unknown path')}")
                return {'status': 'skipped_load_error'}

            image_path = Path(image.filename) # Get original path from PIL image loaded by datasets
            city_name = image_path.parent.name
            base_filename = image_path.name

            results = {'status': 'processed'}
            for prompt, env_var_name in zip(environment_variation_prompts, environment_variation_names):
                output_path = output_dir / env_var_name / city_name / base_filename
                # Skip if already processed
                if output_path.exists():
                    # print(f"Skipping existing: {output_path}")
                    continue

                # Run edit potentially on CPU or GPU
                edited_image = run_edit(
                    pipe_edit,
                    image,
                    prompt + ", photo-realistic", # Add style modifier
                    guidance_scale=args.guidance_scale,
                    steps=args.steps,
                    resolution=args.resolution
                )
                if edited_image:
                    edited_image.save(output_path)
                else:
                    print(f"Warning: Failed to edit image {base_filename} with prompt '{prompt}'")
                    results[f'error_{env_var_name}'] = 'edit_failed'


            return results # Return status or details if needed
        except Exception as e:
            print(f"Error processing {sample.get('image', {}).get('filename', 'unknown image')}: {e}")
            import traceback
            traceback.print_exc()
            # Add filename to error status for better debugging
            filename = "unknown_image"
            if isinstance(sample.get('image'), Image.Image) and hasattr(sample['image'], 'filename'):
                filename = Path(sample['image'].filename).name
            return {'status': 'error', 'error_message': str(e), 'filename': filename}


    # --- Process Data ---
    print(f"Starting image editing process with batch size {args.batch_size}...")
    # Use map for potential parallelism (though GPU limits might make batch_size=1 necessary)
    # Consider num_proc > 1 if CPU pre/post-processing is a bottleneck and batch_size=1
    validation_dataset.map(
        edit_environment_variation,
        batch_size=args.batch_size,
        # num_proc=4 # Optional: Adjust based on CPU cores and IO
    )

    print("Image editing complete.")
    print(f"Edited images saved to: {output_dir}")

if __name__ == "__main__":
    main()
