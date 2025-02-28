import os
# Force Qt to use the xcb plugin (avoid Wayland issues)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import random
import cv2
import torch
import numpy as np
from PIL import Image

# Base directory (location of this script)
base_dir = os.path.dirname(os.path.realpath(__file__))

# Input folder with .jpg images and create output folder for labelled images
input_folder = os.path.join(base_dir, "image_folder")
jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
if not jpg_files:
    raise ValueError("No .jpg files found in 'image_folder'. Check your folder!")

# Ask how many images to process
num_images = int(input("How many images do you want to synthetically label? "))
if num_images > len(jpg_files):
    print("Requested number exceeds available images. Processing all images.")
    selected_files = jpg_files
else:
    selected_files = random.sample(jpg_files, num_images)

output_folder = os.path.join(base_dir, "image_folder_labelled")
os.makedirs(output_folder, exist_ok=True)

# Load the DPT_Large MiDaS model (better for indoor scenes)
print("Loading MiDaS model, please wait...")
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Load the corresponding transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# First pass: compute depth maps and record global min and max
global_min = float('inf')
global_max = -float('inf')
images_data = []  # list of tuples: (filename, original image, depth map)

for filename in selected_files:
    print(f"Processing image: {filename}")
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Unable to read {filename}. Skipping...")
        continue
    # Keep original image for display (BGR)
    original_img = img.copy()
    # Convert to RGB then to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # The transform expects a NumPy array
    input_tensor = transform(np.array(img_pil)).to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        prediction = midas(input_tensor)
        # Resize the depth map to the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    
    # Update global min and max values
    global_min = min(global_min, depth_map.min())
    global_max = max(global_max, depth_map.max())
    
    images_data.append((filename, original_img, depth_map))

print(f"Global depth range: min={global_min:.3f}, max={global_max:.3f}")

# Second pass: normalize depth maps using the global min and max, then display & save
for (filename, original_img, depth_map) in images_data:
    # Normalize using the global range for consistent scaling
    depth_norm = (depth_map - global_min) / (global_max - global_min)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    
    base_name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{base_name}_labelled.jpg")
    cv2.imwrite(output_path, depth_colored)
    print(f"Saved labelled image to: {output_path}")
    
    # Display the original and labelled images side-by-side for verification
    combined = cv2.hconcat([original_img, depth_colored])
    cv2.imshow("Original (Left) vs Labelled (Right) - Press any key to continue", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

print("All images processed and labelled!")
