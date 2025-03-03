#!/usr/bin/env python3

import os
import cv2
import numpy as np
import json
from tqdm import tqdm

def compute_safety_scores(depth_img, grid_width, grid_height,
                          close_threshold=50, mid_threshold=100, far_threshold=150,
                          morph_kernel_size=5,
                          multi_scale_kernel_sizes=(5, 15),
                          dt_weight=1.0, cov_weight=1.0, min_weight=1.0, conn_weight=1.0):
    """
    Full pipeline to compute an (grid_height x grid_width) safety grid from a grayscale depth image.
    
    Pipeline steps:
      1. Multi-thresholding: Compute three binary masks (close, mid, far) where pixels below a threshold are obstacles.
      2. Morphological cleaning: Apply a closing operation to each mask.
      3. Connected Component Analysis: On the close mask, compute connected components and record each component’s min depth.
      4. Coverage: In each grid cell, compute the fraction of pixels that are obstacles in the close mask.
      5. Distance Transform: Compute a normalized distance transform using the union mask from the far threshold.
      6. Multi-scale Min-Depth: Compute local minimum depth maps (via erosion) at multiple scales and average them.
      7. For each grid cell, extract features:
             f_dt   = average normalized distance transform (higher means more free space)
             f_cov  = safe coverage factor = 1 - (obstacle fraction) (higher is safer)
             f_min  = average normalized multi-scale minimum depth (higher means farther away)
             f_conn = worst-case (lowest) normalized min depth from connected obstacles overlapping the cell (if none, 1.0)
      8. Fuse these features using weighted sum to yield a final safety score in [0,1].
    """
    h, w = depth_img.shape

    # --- Step 1: Multi-thresholding ---
    # Lower pixel value indicates closer objects.
    _, mask_close = cv2.threshold(depth_img, close_threshold, 255, cv2.THRESH_BINARY_INV)
    _, mask_mid   = cv2.threshold(depth_img, mid_threshold,   255, cv2.THRESH_BINARY_INV)
    _, mask_far   = cv2.threshold(depth_img, far_threshold,   255, cv2.THRESH_BINARY_INV)

    # --- Step 2: Morphological cleaning ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_CLOSE, kernel)
    mask_mid   = cv2.morphologyEx(mask_mid, cv2.MORPH_CLOSE, kernel)
    mask_far   = cv2.morphologyEx(mask_far, cv2.MORPH_CLOSE, kernel)

    # --- Step 3: Connected Component Analysis on the close mask ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_close, connectivity=8)
    comp_min_depth = {}
    for label in range(1, num_labels):
        # stats[label]: [x, y, width, height, area]
        x, y, w_comp, h_comp, area = stats[label]
        comp_mask = (labels == label)
        min_val = np.min(depth_img[comp_mask])
        # Normalize: 0 means very close (danger), 1 means far (safe)
        comp_min_depth[label] = min_val / 255.0

    # --- Step 4: Distance Transform ---
    # Use the far mask as a union mask (most inclusive).
    free_mask = cv2.bitwise_not(mask_far)
    dist_transform = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    dt_max = np.max(dist_transform)
    if dt_max > 0:
        dist_transform_norm = dist_transform / dt_max
    else:
        dist_transform_norm = dist_transform

    # --- Step 5: Multi-scale Min-Depth using Erosion ---
    depth_float = depth_img.astype(np.float32)
    multi_scale_min_maps = []
    for k_size in multi_scale_kernel_sizes:
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        eroded = cv2.erode(depth_float, kernel_rect)
        # Normalize: higher values mean farther away (safer)
        eroded_norm = eroded / 255.0
        multi_scale_min_maps.append(eroded_norm)
    multi_scale_min = np.mean(np.stack(multi_scale_min_maps, axis=-1), axis=-1)

    # --- Step 6: Grid Aggregation ---
    cell_height = h // grid_height
    cell_width  = w // grid_width
    safety_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    for gy in range(grid_height):
        for gx in range(grid_width):
            y_start = gy * cell_height
            y_end   = (gy+1) * cell_height if gy < grid_height - 1 else h
            x_start = gx * cell_width
            x_end   = (gx+1) * cell_width if gx < grid_width - 1 else w

            # --- Feature f_dt: Distance Transform ---
            dt_cell = dist_transform_norm[y_start:y_end, x_start:x_end]
            f_dt = np.mean(dt_cell)
            
            # --- Feature f_cov: Coverage (from mask_close) ---
            cell_mask = mask_close[y_start:y_end, x_start:x_end]
            obstacle_fraction = np.count_nonzero(cell_mask) / (cell_mask.size)
            f_cov = 1.0 - obstacle_fraction

            # --- Feature f_min: Multi-scale Min-Depth ---
            ms_cell = multi_scale_min[y_start:y_end, x_start:x_end]
            f_min = np.mean(ms_cell)

            # --- Feature f_conn: Connected Component ---
            overlapping_scores = []
            cell_box = (x_start, y_start, x_end, y_end)
            for label in range(1, num_labels):
                comp_x, comp_y, comp_w, comp_h, _ = stats[label]
                comp_box = (comp_x, comp_y, comp_x + comp_w, comp_y + comp_h)
                # Check for overlap
                if (comp_box[0] < cell_box[2] and comp_box[2] > cell_box[0] and
                    comp_box[1] < cell_box[3] and comp_box[3] > cell_box[1]):
                    overlapping_scores.append(comp_min_depth[label])
            f_conn = min(overlapping_scores) if overlapping_scores else 1.0

            # --- Fuse features with weights ---
            total_weight = dt_weight + cov_weight + min_weight + conn_weight
            raw_score = (dt_weight * f_dt + cov_weight * f_cov + min_weight * f_min + conn_weight * f_conn) / total_weight

            safety_grid[gy, gx] = max(0.0, min(1.0, raw_score))
    
    return safety_grid

def overlay_safety_grid(base_img, safety_grid, alpha=0.4):
    """
    Overlays the safety grid onto a base image.
    Grid values in [0,1] are mapped: 0 -> red (unsafe) and 1 -> green (safe).
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
            y_end   = (gy+1) * cell_h if gy < grid_height - 1 else h
            x_start = gx * cell_w
            x_end   = (gx+1) * cell_w if gx < grid_width - 1 else w

            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)

    blended = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)
    return blended

def compose_output_image(original_img, original_overlay, depth_img, depth_overlay, safety_grid):
    """
    Creates a composite image containing:
      - Top-left: original image.
      - Bottom-left: original image with safety grid overlay.
      - Top-center: depth image.
      - Bottom-center: depth image with safety grid overlay.
      - Right: a table listing numerical safety scores per grid cell.
    """
    h, w, _ = original_img.shape
    montage_w = w * 3
    montage_h = h * 2
    composite = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 255

    composite[0:h, 0:w] = original_img
    composite[h:2*h, 0:w] = original_overlay
    composite[0:h, w:2*w] = depth_img
    composite[h:2*h, w:2*w] = depth_overlay

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
    parser.add_argument("--images_folder", type=str, default="images_folder",
                        help="Folder containing original RGB images named <long number>.jpg")
    parser.add_argument("--depth_folder", type=str, default="synthetic_depth_maps",
                        help="Folder containing greyscale depth images named depth_<long number>.jpg")
    parser.add_argument("--output_folder", type=str, default="labelled_images_oai_extended",
                        help="Folder to save the output composite images and JSON summary")
    parser.add_argument("--grid_width", type=int, default=9, help="Number of grid columns")
    parser.add_argument("--grid_height", type=int, default=9, help="Number of grid rows")
    
    # Full pipeline hyperparameters
    parser.add_argument("--close_threshold", type=int, default=50, help="Threshold for 'close' obstacles (0-255)")
    parser.add_argument("--mid_threshold", type=int, default=100, help="Threshold for 'mid' obstacles (0-255)")
    parser.add_argument("--far_threshold", type=int, default=150, help="Threshold for 'far' obstacles (0-255)")
    parser.add_argument("--morph_kernel_size", type=int, default=5, help="Kernel size for morphological operations")
    parser.add_argument("--multi_scale_kernel_sizes", nargs="+", type=int, default=[5, 15],
                        help="Kernel sizes for multi-scale erosion")
    parser.add_argument("--dt_weight", type=float, default=1.0, help="Weight for distance transform feature")
    parser.add_argument("--cov_weight", type=float, default=1.0, help="Weight for coverage feature")
    parser.add_argument("--min_weight", type=float, default=1.0, help="Weight for multi-scale min-depth feature")
    parser.add_argument("--conn_weight", type=float, default=1.0, help="Weight for connected component feature")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(args.images_folder) if f.lower().endswith(".jpg")]
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
            print(f"Could not read {filename} or {depth_filename}, skipping.")
            continue

        # Rotate images 90° CCW to fix orientation.
        original_bgr = cv2.rotate(original_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_gray = cv2.rotate(depth_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
            mid_threshold=args.mid_threshold,
            far_threshold=args.far_threshold,
            morph_kernel_size=args.morph_kernel_size,
            multi_scale_kernel_sizes=args.multi_scale_kernel_sizes,
            dt_weight=args.dt_weight,
            cov_weight=args.cov_weight,
            min_weight=args.min_weight,
            conn_weight=args.conn_weight
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

        results_dict[base_name] = {
            "grid_width": args.grid_width,
            "grid_height": args.grid_height,
            "scores": safety_grid.tolist()
        }

    json_path = os.path.join(args.output_folder, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Done! Labelled images and 'results.json' saved in {args.output_folder}")

if __name__ == "__main__":
    main()
