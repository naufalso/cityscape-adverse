import os
import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import random

# Define paths
original_path = "data/cityscapes/leftImg8bit/val"
ground_truth_path = "data/cityscapes/gtFine/val"
modified_path = "data/cityscapes-adverse-hd"
output_folder = "data/cityscapes-adverse-hd-filter/"
os.makedirs(output_folder, exist_ok=True)

# Global variable to store initial labeled counts
first_init_labeled_counts = None

# Function to get already labeled images
def get_labeled_images(modification_type):
    accepted_file = os.path.join(output_folder, f"{modification_type}_accepted.txt")
    rejected_file = os.path.join(output_folder, f"{modification_type}_rejected.txt")
    
    labeled_images = set()
    
    if os.path.exists(accepted_file):
        with open(accepted_file, 'r') as f:
            labeled_images.update(f.read().splitlines())
    
    if os.path.exists(rejected_file):
        with open(rejected_file, 'r') as f:
            labeled_images.update(f.read().splitlines())
    
    return labeled_images

# Function to get image pairs
def get_image_pairs(modification_type):
    print(f"Getting image pairs for modification type: {modification_type}")
    labeled_images = get_labeled_images(modification_type)
    city_names = os.listdir(os.path.join(modified_path, modification_type))
    image_pairs = []
    for city in city_names:
        mod_images = os.listdir(os.path.join(modified_path, modification_type, city))
        for img in mod_images:
            mod_img_path = os.path.join(modified_path, modification_type, city, img)
            if mod_img_path in labeled_images:
                continue
            orig_img = os.path.join(original_path, city, img)
            mod_img = mod_img_path
            if os.path.exists(orig_img) and os.path.exists(mod_img):
                image_pairs.append((orig_img, mod_img))
    return image_pairs

# Precompute image pairs for all modification types
def precompute_image_pairs():
    modification_types = os.listdir(modified_path)
    all_image_pairs = {}
    for mod_type in modification_types:
        all_image_pairs[mod_type] = get_image_pairs(mod_type)
        random.shuffle(all_image_pairs[mod_type])  # Shuffle the list to randomize
    return all_image_pairs

# Function to load images
def load_images(image_pair):
    orig_img = Image.open(image_pair[0])
    mod_img = Image.open(image_pair[1])
    return orig_img, mod_img

# Function to handle accept/reject actions
def handle_action(action, image_pair, modification_type, labeled_count):
    if action == "accept":
        result = f"Accepted: {os.path.basename(image_pair[1])}"
        with open(os.path.join(output_folder, f"{modification_type}_accepted.txt"), "a") as f:
            f.write(f"{image_pair[1]}\n")
    else:
        result = f"Rejected: {os.path.basename(image_pair[1])}"
        with open(os.path.join(output_folder, f"{modification_type}_rejected.txt"), "a") as f:
            f.write(f"{image_pair[1]}\n")
    labeled_count += 1
    return result, labeled_count

# Create a Gradio interface
def create_interface():
    global first_init_labeled_counts
    all_image_pairs = precompute_image_pairs()
    modification_types = list(all_image_pairs.keys())
    labeled_counts = {mod_type: len(get_labeled_images(mod_type)) for mod_type in modification_types}
    first_init_labeled_counts = labeled_counts.copy()  # Initialize first_init_labeled_counts

    def next_image(modification_type, orig_image_path, mod_image_path, action=None):
        nonlocal labeled_counts
        global first_init_labeled_counts
        mod_type = modification_type

        unlabeled_images = [pair for pair in all_image_pairs[mod_type] if pair[1] not in get_labeled_images(mod_type)]
        
        if unlabeled_images:
            image_pair_index = random.randint(0, len(unlabeled_images) - 1)
            if action is not None:
                result, labeled_counts[mod_type] = handle_action(action, [orig_image_path, mod_image_path], mod_type, labeled_counts[mod_type])
                # Recompute unlabeled_images after handling action
                unlabeled_images = [pair for pair in all_image_pairs[mod_type] if pair[1] not in get_labeled_images(mod_type)]
            
            if unlabeled_images:
                image_pair_index = random.randint(0, len(unlabeled_images) - 1)
                orig_img_path, mod_img_path = unlabeled_images[image_pair_index]
                orig_img, mod_img = load_images(unlabeled_images[image_pair_index])
                progress = f"{labeled_counts[mod_type]} / {len(all_image_pairs[mod_type]) + first_init_labeled_counts[mod_type]}"
                return orig_img, mod_img, orig_img_path, mod_img_path, progress
            else:
                return None, None, None, None, "All images have been labeled for the selected modification type."
        else:
            return None, None, None, None, "All images have been labeled for the selected modification type."

    with gr.Blocks() as demo:
        gr.Markdown("# Image Verification")
        modification_type_dropdown = gr.Dropdown(label="Modification Type", choices=modification_types)
        with gr.Row():
            orig_image = gr.Image(label="Original Image")
            mod_image = gr.Image(label="Modified Image")
            
        progress_label = gr.Label(label="Progress")
        orig_image_path = gr.Text(label="Original Path")
        mod_image_path = gr.Text(label="Modified Path")

        # result = gr.Textbox(label="Result")
        with gr.Row():
            accept_button = gr.Button("Accept")
            reject_button = gr.Button("Reject")

        modification_type_dropdown.change(next_image, inputs=[modification_type_dropdown, orig_image_path, mod_image_path], outputs=[orig_image, mod_image, orig_image_path, mod_image_path, progress_label])
        accept_button.click(next_image, inputs=[modification_type_dropdown, orig_image_path, mod_image_path, gr.Text(value="accept", visible=False)], outputs=[orig_image, mod_image, orig_image_path, mod_image_path, progress_label])
        reject_button.click(next_image, inputs=[modification_type_dropdown, orig_image_path, mod_image_path, gr.Text(value="reject", visible=False)], outputs=[orig_image, mod_image, orig_image_path, mod_image_path, progress_label])

    return demo


if __name__ == "__main__":
    # Run the interface
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=6006, share=True)