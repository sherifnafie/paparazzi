import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

def process_image(image_path, lambda_weight):  # λ controls center pull
    start_time = time.time()
    image = cv2.imread(image_path)  
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Rotate image to correct orientation
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Convert to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)  # Adjust thresholds as needed
    
    # Define YUV color ranges for detection   
    yellow_lower = np.array([180, 20, 120], dtype=np.uint8)
    yellow_upper = np.array([255, 100, 255], dtype=np.uint8)
    pink_lower = np.array([110, 10, 150], dtype=np.uint8)
    pink_upper = np.array([255, 150, 255], dtype=np.uint8)
    
    # Create masks for yellow and pink
    pink_mask = cv2.inRange(image_yuv, pink_lower, pink_upper)
    yellow_mask = cv2.inRange(image_yuv, yellow_lower, yellow_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, pink_mask)

    # Apply morphological operations
    kernel = np.ones((6, 6), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # Find contours for detected areas
    min_object_size = 500
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output image for detected colors
    detected_colors = np.zeros_like(image)

    for contour in contours:
        if cv2.contourArea(contour) > min_object_size:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Assign a random color to the contour
            random_color = [random.randint(0, 255) for _ in range(3)]
            cv2.drawContours(detected_colors, [contour], -1, random_color, thickness=cv2.FILLED)

    # Create zone heatmap (3 rows × 5 columns)
    rows, cols = 3, 5
    h, w = combined_mask.shape
    zone_h, zone_w = h // rows, w // cols
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    bottom_row_coverage = np.zeros(cols, dtype=np.float32)  # Store coverage per column (only bottom row)

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * zone_w, r * zone_h
            x2, y2 = (c + 1) * zone_w, (r + 1) * zone_h

            # Extract the zone
            zone_mask = combined_mask[y1:y2, x1:x2]
            zone_area = zone_mask.size
            white_pixels = cv2.countNonZero(zone_mask)
            coverage_ratio = white_pixels / zone_area  # Percentage of coverage

            # Store coverage for bottom row only
            if r == rows - 1:  # Only store the last row (bottom row)
                bottom_row_coverage[c] = coverage_ratio  

            # Color scale from green (0%) to red (100%)
            color = (0, int((1 - coverage_ratio) * 255), int(coverage_ratio * 255))  # BGR

            # Fill the heatmap zone
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

            # Draw zone borders in white
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

    # Blend heatmap onto the processed image with 20% transparency
    overlay = cv2.addWeighted(image, 1.0, heatmap, 0.5, 0)

    # Determine safest column from bottom row (least coverage)
    safest_column = np.argmin(bottom_row_coverage)  
    angle = np.interp(safest_column, [0, cols - 1], [90, -90])  # Convert column index to angle

    # **Apply Weighting Factor to Pull Toward Center**
    adjusted_angle = (1 - lambda_weight) * angle  # Weighted angle calculation

    # Calculate dot position (now inside bottom row)
    dot_x = int(np.interp(adjusted_angle, [-90, 90], [w, 0]))  # Map angle to x-position
    dot_y = h - (zone_h // 2)  # Place in the middle of the bottom row

    # Draw red dot (direction visualization)
    cv2.circle(overlay, (dot_x, dot_y), radius=15, color=(0, 0, 255), thickness=-1)

    execution_time = time.time() - start_time
    print(f"Processed {image_path} in {execution_time:.4f} sec - Angle: {angle:.2f}° → Adjusted: {adjusted_angle:.2f}°")
    
    # Display results in a 2x2 grid
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(pink_mask, cmap='gray')
    plt.title("Pink Mask")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Processed + Heatmap + Direction ({adjusted_angle:.1f}°)")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(detected_colors, cv2.COLOR_BGR2RGB))
    plt.title("Detected Colors (Random Contour Colors)")
    plt.axis("off")
    
    plt.tight_layout()  # Ensures proper spacing between plots
    
    plt.show()
    plt.pause(0)

# Process images in folder
image_folder = "./image_folder3"
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

if not image_files:
    print("No images found in the folder!")

plt.ion()
for image_file in image_files:
    process_image(image_file, lambda_weight=0.7)  # Set λ (default = 0.5)
plt.ioff()
plt.show()
