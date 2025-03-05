#!/usr/bin/env python3
#python .\synthetic_labeller_oai.py --depth_folder depth_predictions_DAV2 --output_folder labelled_images_oai_DAV2 --invert_depth

import os
import cv2
import numpy as np
import json
from tqdm import tqdm

def compute_safety_scores(depth_img, grid_width, grid_height,
                          close_threshold=50,   # example param in [0..255]
                          morph_kernel_size=5,
                          multi_scale_kernel_sizes=(5, 15)):
    """
    Given a single-channel depth image (grayscale),
    compute an (grid_height x grid_width) array of "safety scores".
    
    This function demonstrates a simplified version of the pipeline:
      1) Threshold and morphological cleanup
      2) Distance transform
      3) Multi-scale min-depth checks
      4) Aggregate sector-wise features into a final safety score
    
    Returns: safety_grid (2D numpy array of shape [grid_height, grid_width])
    """
    h, w = depth_img.shape

    # 1) Basic morphological cleaning on a thresholded mask
    ret, obstacle_mask = cv2.threshold(depth_img, close_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

    # 2) Distance transform on the cleaned mask
    free_mask = cv2.bitwise_not(obstacle_mask)
    dist_transform = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    dist_transform_norm = dist_transform / (dist_transform.max() + 1e-5)

    # 3) Multi-scale checks:
    depth_float = depth_img.astype(np.float32)
    multi_scale_features = []
    for k_size in multi_scale_kernel_sizes:
        avg = cv2.boxFilter(depth_float, ddepth=-1, ksize=(k_size, k_size))
        avg_norm = avg / 255.0
        multi_scale_features.append(avg_norm)
    
    combined_multi_scale = np.mean(np.stack(multi_scale_features, axis=-1), axis=-1)

    # 4) Partition the image into grid cells and compute a final safety score
    cell_height = h // grid_height
    cell_width  = w // grid_width
    safety_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    for gy in range(grid_height):
        for gx in range(grid_width):
            y_start = gy * cell_height
            y_end   = (gy+1) * cell_height if gy < grid_height-1 else h
            x_start = gx * cell_width
            x_end   = (gx+1) * cell_width if gx < grid_width-1 else w

            sector_dist = dist_transform_norm[y_start:y_end, x_start:x_end]
            sector_depth_avg = combined_multi_scale[y_start:y_end, x_start:x_end]

            dist_score = np.mean(sector_dist)
            depth_score = np.mean(sector_depth_avg)

            raw_score = 0.5 * dist_score + 0.5 * depth_score
            safety_grid[gy, gx] = max(0.0, min(1.0, raw_score))

    return safety_grid

def overlay_safety_grid(base_img, safety_grid, alpha=0.4):
    """
    Overlay the safety_grid onto the base image.
    safety_grid is assumed normalized in [0..1], where 0=least safe (red) and 1=safest (green).
    base_img can be color (H,W,3) in uint8.
    Returns a copy of base_img overlaid with color blocks.
    """
    overlay = base_img.copy()
    h, w, _ = overlay.shape
    grid_height, grid_width = safety_grid.shape
    cell_h = h // grid_height
    cell_w = w // grid_width

    for gy in range(grid_height):
        for gx in range(grid_width):
            val = safety_grid[gy, gx]
            r = int((1.0 - val) * 255)
            g = int(val * 255)
            b = 0
            color = (b, g, r)  # OpenCV uses BGR

            y_start = gy * cell_h
            y_end   = (gy+1) * cell_h if gy < grid_height-1 else h
            x_start = gx * cell_w
            x_end   = (gx+1) * cell_w if gx < grid_width-1 else w

            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)

    blended = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)
    return blended

def compose_output_image(original_img, original_overlay, depth_img, depth_overlay, safety_grid):
    """
    Create a single large image that shows:
      - top-left: original_img
      - bottom-left: original_overlay
      - top-center: depth_img
      - bottom-center: depth_overlay
      - on the right: a textual table of the safety_grid values
    
    Returns: a composite BGR image (np.uint8).
    """
    h, w, _ = original_img.shape
    montage_w = w * 3
    montage_h = h * 2
    composite = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 255

    composite[0:h, 0:w] = original_img
    composite[h:2*h, 0:w] = original_overlay
    composite[0:h, w:2*w] = depth_img
    composite[h:2*h, w:2*w] = depth_overlay

    table_x_start = 2 * w
    text_region_x0 = 2 * w
    text_region_y0 = 0
    cell_spacing = 30
    grid_height, grid_width = safety_grid.shape
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(composite, "Safety Grid Scores:", (text_region_x0 + 10, text_region_y0 + 25),
                font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    y_cursor = text_region_y0 + 50
    for gy in range(grid_height):
        row_values = safety_grid[gy]
        text_line = f"Row {gy}: " + ", ".join(f"{v:.2f}" for v in row_values)
        cv2.putText(composite, text_line, (text_region_x0 + 10, y_cursor),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        y_cursor += cell_spacing

    return composite

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type=str, default= r"C:\Users\beren\Documents\paparazzi\playground\videocap_simulation_round1",
                        help="Path to the folder containing original RGB images named <long number>.jpg")
    parser.add_argument("--depth_folder", type=str, default=r"C:\Users\beren\Documents\paparazzi\playground\depth_predictions_DAV2",
                        help="Path to the folder containing greyscale depth images named depth_<long number>.jpg")
    parser.add_argument("--output_folder", type=str, default=r"C:\Users\beren\Documents\paparazzi\playground\labelled_images_oai",
                        help="Folder where the output composite images and JSON file will be saved")
    parser.add_argument("--grid_width", type=int, default=16, help="Number of grid columns")
    parser.add_argument("--grid_height", type=int, default=6, help="Number of grid rows")
    parser.add_argument("--close_threshold", type=int, default=60,
                        help="Threshold in [0..255] for what is considered 'close obstacle'")
    parser.add_argument("--morph_kernel_size", type=int, default=5,
                        help="Kernel size for morphological closure")
    parser.add_argument("--multi_scale_kernel_sizes", nargs="+", type=int, default=[15, 15],
                        help="Sizes of the multi-scale kernels (box filter or min filter)")
    parser.add_argument("--invert_depth", action="store_true",
                        help="Invert depth images (if light pixels are close, dark pixels are far)")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    image_files = []
    for f in os.listdir(args.images_folder):
        if f.lower().endswith(".jpg"):
            image_files.append(f)
    image_files.sort()

    results_dict = {}

    for filename in tqdm(image_files, desc="Processing"):
        base_name = os.path.splitext(filename)[0]
        rgb_path = os.path.join(args.images_folder, filename)
        depth_filename = f"depth_{base_name}.jpg"
        depth_path = os.path.join(args.depth_folder, depth_filename)

        if not os.path.isfile(depth_path):
            print(f"No matching depth file for {filename}, skipping.")
            continue

        original_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if original_bgr is None or depth_gray is None:
            print(f"Could not read {filename} or {depth_filename}. Skipping.")
            continue

        # # Rotate images 90 degrees counterclockwise to fix orientation.
        # original_bgr = cv2.rotate(original_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # depth_gray = cv2.rotate(depth_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Invert depth image if flag is set (light pixels are close, dark pixels are far)
        if args.invert_depth:
            depth_gray = cv2.bitwise_not(depth_gray)

        h1, w1, _ = original_bgr.shape
        h2, w2 = depth_gray.shape
        if (h1 != h2) or (w1 != w2):
            print(f"Size mismatch between {filename} and {depth_filename}, resizing depth to match.")
            depth_gray = cv2.resize(depth_gray, (w1, h1), interpolation=cv2.INTER_NEAREST)

        safety_grid = compute_safety_scores(
            depth_gray,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            close_threshold=args.close_threshold,
            morph_kernel_size=args.morph_kernel_size,
            multi_scale_kernel_sizes=args.multi_scale_kernel_sizes
        )

        overlay_original = overlay_safety_grid(original_bgr, safety_grid, alpha=0.4)
        depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        overlay_depth = overlay_safety_grid(depth_bgr, safety_grid, alpha=0.4)

        composite = compose_output_image(original_bgr, overlay_original,
                                         depth_bgr, overlay_depth,
                                         safety_grid)
        
        out_filename = f"labelled_{base_name}.jpg"
        out_path = os.path.join(args.output_folder, out_filename)
        cv2.imwrite(out_path, composite)

        scores_list = safety_grid.tolist()
        results_dict[base_name] = {
            "grid_width":  args.grid_width,
            "grid_height": args.grid_height,
            "scores": scores_list
        }

    json_path = os.path.join(args.output_folder, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Done! Labelled images and 'results.json' saved in {args.output_folder}")

if __name__ == "__main__":
    main()
