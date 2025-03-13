
# =============================================================================
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import time
# 
# # Define color ranges in YUV space with enable/disable option
# color_settings = {
#     "yellow": {"lower": np.array([150, 0, 100], dtype=np.uint8),
#                "upper": np.array([255, 120, 255], dtype=np.uint8),
#                "enabled": True},  # Set to False to disable
# 
#     "pink": {"lower": np.array([110, 10, 150], "uint8"),
#              "upper": np.array([255, 150, 255], "uint8"),
#              "enabled": True},
# 
#     "blue": {"lower": np.array([100, 0, 0], "uint8"),
#              "upper": np.array([255, 120, 120], "uint8"),
#              "enabled": False},
# 
#     "green": {"lower": np.array([0, 50, 0], "uint8"),
#               "upper": np.array([100, 255, 100], "uint8"),
#               "enabled": False},
# 
#     "red": {"lower": np.array([0, 0, 100], "uint8"),
#             "upper": np.array([100, 100, 255], "uint8"),
#             "enabled": False},
# }
# 
# # Store last 10 direction changes
# direction_changes = []
# 
# def process_image(image_path, lambda_weight=0.5):
#     start_time = time.time()
#     image = cv2.imread(image_file)
#     if image is None:
#         print(f"Error: Unable to load {image_file}")
#         return
#     
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#     
#     # Create a dictionary for individual color masks
#     individual_masks = {}
#     combined_mask = np.zeros(image_yuv.shape[:2], dtype=np.uint8)
# 
#     # Apply enabled color masks
#     for color_name, data in color_settings.items():
#         if data["enabled"]:
#             mask = cv2.inRange(image_yuv, data["lower"], data["upper"])
#             individual_masks[color_name] = mask
#             combined_mask = cv2.bitwise_or(combined_mask, mask)
# 
#     # Morphological operations to clean up masks
#     kernel = np.ones((6, 6), np.uint8)
#     mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.dilate(mask, kernel, iterations=2)
# 
#     # Find contours
#     min_object_size = 500  # Minimum area to consider as an object
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     
#     # Draw bounding boxes and color segmentation
#     detected_color_areas = np.zeros_like(image)
#     random_colors = {}
# 
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > min_object_size:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
# 
#             # Assign random color if not already assigned
#             index = len(random_colors)  # Assign unique index for each contour
#             if index not in random_colors:
#                 random_colors[index] = np.random.randint(0, 255, (1, 3), dtype=np.uint8)[0]
#             
#             cv2.drawContours(detected_color_areas, [contour], -1, random_colors[index].tolist(), thickness=cv2.FILLED)
# 
# 
#     # Compute safe direction using zone analysis
#     height, width = mask.shape
#     rows, cols = 3, 5
#     zone_height = height // 3
#     zone_width = width // 5
#     min_density = float("inf")
#     best_x = width // 2
#     best_y = height // 2
# 
#     for row in range(3):
#         for col in range(5):
#             x1, y1 = col * width // 5, row * height // 3
#             x2, y2 = x1 + width // 5, y1 + height // 3
#             zone_mask = mask[y1:y2, x1:x2]
#             zone_area = np.sum(zone_mask) / 255
#             total_area = (x2 - x1) * (y2 - y1)
#             density = zone_mask.sum() / (255 * (width // 5) * (height // 3))
# 
#             # Find the safest direction (lowest density)
#             if row == 2:  # Bottom row (ideal direction)
#                 direction = 0
#             else:
#                 direction = (col - 2) * 45  # Map column to angle: -90, -45, 0, 45, 90
# 
#             # Adjust weighting towards safer zones
#             direction_weight = (1 - lambda_weight) * direction
#             direction_changes.append(direction)
# 
#             if len(direction_changes) > 10:
#                 direction_changes.pop(0)  # Keep only the last 10 values
# 
#     # Compute average direction over last 10 frames
#     avg_direction = sum(direction_changes) / len(direction_changes)
# 
#     # Compute center point of image
#     center_x, center_y = width // 2, height // 2
#     arrow_length = 50
# 
#     # Compute direction in radians and find new arrow endpoint
#     angle_rad = np.deg2rad(avg_direction)
#     arrow_end_x = int(center_x + arrow_length * np.cos(angle_rad))
#     arrow_y = center_y + int(arrow_length * np.sin(angle_rad))
# 
#     # Draw red dot for direction change
#     cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)
#     cv2.line(image, (center_x, center_y), (arrow_length, center_y - int(arrow_length * np.tan(angle_rad))), (0, 0, 255), 3)
# 
#     # Display images in two rows
#     plt.figure(figsize=(15, 10))
# 
#     # First row: Processed image with zones
#     plt.subplot(2, 3, 1)
#     plt.imshow(image)
#     plt.title("Processed Image")
#     plt.axis("off")
# 
#     plt.subplot(2, 3, 2)
#     plt.imshow(mask, cmap="gray")
#     plt.title("Combined Mask")
#     plt.axis("off")
# 
#     plt.subplot(2, 3, 3)
#     plt.imshow(detected_color_areas)
#     plt.title("Detected Colors")
#     plt.axis("off")
# 
#     # Show all enabled individual masks
#     for i, (color_name, mask) in enumerate(individual_masks.items()):
#         plt.subplot(2, 3, i + 4)
#         plt.imshow(mask, cmap="gray")
#         plt.title(f"{color_name} Mask")
#         plt.axis("off")
# 
#     plt.show()
#     plt.pause(0.2)
# 
#     print(f"Processed {image_file}, Avg Direction Change: {avg_direction:.2f}°")
# 
# 
# # Process images in folder
# image_folder = "./image_folder2"
# image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
# plt.ion()
# for image_file in image_files:
#     process_image(image_file, lambda_weight=0.5)
# plt.show()
# 
# =============================================================================


# =============================================================================
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os
# import random
# from collections import deque  # Stores last 10 direction changes
# 
# # Store last 10 direction values
# direction_history = deque(maxlen=10)
# 
# def process_image(image_path, lambda_weight=0.5):  # λ controls center pull
#     start_time = time.time()
#     image = cv2.imread(image_path)  
#     if image is None:
#         print(f"Error: Unable to read {image_path}")
#         return
#     
#     # Rotate image to correct orientation
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     
#     # Convert to YUV color space
#     image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#     
#     # Define YUV color ranges for detection
#     color_ranges = {
#         "Pink":   (np.array([110, 10, 150], dtype=np.uint8), np.array([255, 150, 255], dtype=np.uint8)),
#         "Yellow": (np.array([150, 0, 100], dtype=np.uint8),  np.array([255, 120, 255], dtype=np.uint8)),
#         "Blue":   (np.array([0, 50, 100], dtype=np.uint8),   np.array([120, 255, 255], dtype=np.uint8)),
#         "Green":  (np.array([50, 0, 50], dtype=np.uint8),    np.array([255, 120, 100], dtype=np.uint8)),
#         "Orange": (np.array([100, 50, 0], dtype=np.uint8),   np.array([255, 200, 100], dtype=np.uint8))
#     }
#     color_settings = {
#     "yellow": {"lower": np.array([150, 0, 100], dtype=np.uint8),
#                "upper": np.array([255, 120, 255], dtype=np.uint8),
#                "enabled": True},  # Set to False to disable
#                
#     "pink": {"lower": np.array([110, 10, 150], dtype=np.uint8),
#              "upper": np.array([255, 150, 255], dtype=np.uint8),
#              "enabled": True},  # Set to False to disable
#              
#     "blue": {"lower": np.array([50, 0, 0], dtype=np.uint8),
#              "upper": np.array([200, 120, 120], dtype=np.uint8),
#              "enabled": True},  # Set to False to disable
# 
#     "green": {"lower": np.array([0, 50, 0], dtype=np.uint8),
#               "upper": np.array([100, 255, 100], dtype=np.uint8),
#               "enabled": False},  # Disabled by default
# 
#     "red": {"lower": np.array([0, 0, 100], dtype=np.uint8),
#             "upper": np.array([100, 100, 255], dtype=np.uint8),
#             "enabled": False},  # Disabled by default
# }
#     
#     # Create combined mask and color detection map
#     combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     detected_colors = np.zeros_like(image)
#     
#     individual_masks = {}
#     for color_name, data in color_settings.items():
#         if data["enabled"]:
#             mask = cv2.inRange(image_yuv, data["lower"], data["upper"])
#             individual_masks[color_name] = mask  # Store enabled color masks
# 
# 
#     for color_name, (lower, upper) in color_ranges.items():
#         mask = cv2.inRange(image_yuv, lower, upper)
#         combined_mask = cv2.bitwise_or(combined_mask, mask)
# 
#         # Assign a random color to each detected region
#         random_color = [random.randint(0, 255) for _ in range(3)]
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Ignore small noise
#                 cv2.drawContours(detected_colors, [contour], -1, random_color, thickness=cv2.FILLED)
# 
#     # Apply morphological operations to clean up mask
#     kernel = np.ones((6, 6), np.uint8)
#     mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.dilate(mask, kernel, iterations=2)
# 
#     # Find contours for detected areas
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
#     # Draw bounding boxes on the image
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
# 
#     # Create zone heatmap (3 rows × 5 columns)
#     rows, cols = 3, 5
#     h, w = mask.shape
#     zone_h, zone_w = h // rows, w // cols
#     heatmap = np.zeros((h, w, 3), dtype=np.uint8)
#     column_coverage = np.zeros(cols, dtype=np.float32)
# 
#     for r in range(rows):
#         for c in range(cols):
#             x1, y1 = c * zone_w, r * zone_h
#             x2, y2 = (c + 1) * zone_w, (r + 1) * zone_h
# 
#             # Extract the zone
#             zone_mask = mask[y1:y2, x1:x2]
#             zone_area = zone_mask.size
#             white_pixels = cv2.countNonZero(zone_mask)
#             coverage_ratio = white_pixels / zone_area  # Percentage of coverage
# 
#             # Store coverage for column
#             column_coverage[c] += coverage_ratio  
# 
#             # Color scale from green (0%) to red (100%)
#             color = (0, int((1 - coverage_ratio) * 255), int(coverage_ratio * 255))  # BGR
# 
#             # Fill the heatmap zone
#             cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
# 
#             # Draw zone borders in white
#             cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
# 
#     # Blend heatmap onto the processed image with 20% transparency
#     overlay = cv2.addWeighted(image, 1.0, heatmap, 0.2, 0)
# 
#     # Determine safest column (least coverage)
#     safest_column = np.argmin(column_coverage)  
#     angle = np.interp(safest_column, [0, cols - 1], [90, -90])  # Convert column index to angle
# 
#     # **Apply Weighting Factor to Pull Toward Center**
#     adjusted_angle = (1 - lambda_weight) * angle  
# 
#     # **Store in Direction History (Last 10 Values)**
#     direction_history.append(adjusted_angle)
# 
#     # **Compute Rolling Average Over Last 10 Images**
#     avg_angle = np.mean(direction_history)
# 
#     # Calculate dot position
#     dot_x = int(np.interp(avg_angle, [-90, 90], [w, 0]))  # Map angle to x-position
#     dot_y = h // 2  # Center vertically
# 
#     # Draw red dot for direction change
#     cv2.circle(overlay, (dot_x, dot_y), radius=15, color=(0, 0, 255), thickness=-1)
# 
#     execution_time = time.time() - start_time
#     print(f"Processed {image_path} in {execution_time:.4f} sec - Raw: {angle:.2f}° | Adjusted: {adjusted_angle:.2f}° | Avg: {avg_angle:.2f}°")
#     
#     # Display results
#     plt.figure(figsize=(18, 5))
#     
#     plt.subplot(1, 4, 1)
#     plt.imshow(combined_mask, cmap='gray')
#     plt.title("Combined Mask")
#     plt.axis("off")
#     
#     plt.subplot(1, 4, 2)
#     plt.imshow(cv2.cvtColor(detected_colors, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Colors")
#     plt.axis("off")
#     
#     plt.subplot(1, 4, 3)
#     plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     plt.title(f"Processed + Heatmap + Avg Direction ({avg_angle:.2f}°)")
#     plt.axis("off")
#     
#     plt.subplot(1, 4, 4)
#     plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     plt.title("Direction Indicator")
#     plt.axis("off")
#     
#     plt.show()
#     plt.pause(0.2)
# 
# # Process images in folder
# image_folder = "./image_folder2"
# image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
# 
# if not image_files:
#     print("No images found in the folder!")
# 
# plt.ion()
# for image_file in image_files:
#     process_image(image_file, lambda_weight=0.5)  # Set λ (default = 0.5)
# plt.ioff()
# plt.show()
# =============================================================================



import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

def process_image(image_path, lambda_weight=0.5):  # λ controls center pull
    start_time = time.time()
    image = cv2.imread(image_path)  
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Rotate image to correct orientation
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Convert to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Define YUV color ranges for detection
    yellow_lower = np.array([150, 0, 100], dtype=np.uint8)
    yellow_upper = np.array([255, 120, 255], dtype=np.uint8)
    pink_lower = np.array([110, 10, 150], dtype=np.uint8)
    pink_upper = np.array([255, 150, 255], dtype=np.uint8)
    
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
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours for detected areas
    min_object_size = 500
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    h, w = mask.shape
    zone_h, zone_w = h // rows, w // cols
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    column_coverage = np.zeros(cols, dtype=np.float32)  # Store coverage per column

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * zone_w, r * zone_h
            x2, y2 = (c + 1) * zone_w, (r + 1) * zone_h

            # Extract the zone
            zone_mask = mask[y1:y2, x1:x2]
            zone_area = zone_mask.size
            white_pixels = cv2.countNonZero(zone_mask)
            coverage_ratio = white_pixels / zone_area  # Percentage of coverage

            # Store coverage for column
            column_coverage[c] += coverage_ratio  

            # Color scale from green (0%) to red (100%)
            color = (0, int((1 - coverage_ratio) * 255), int(coverage_ratio * 255))  # BGR

            # Fill the heatmap zone
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

            # Draw zone borders in white
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

    # Blend heatmap onto the processed image with 20% transparency
    overlay = cv2.addWeighted(image, 1.0, heatmap, 0.5, 0)

    # Determine safest column (least coverage)
    safest_column = np.argmin(column_coverage)  
    angle = np.interp(safest_column, [0, cols - 1], [90, -90])  # Convert column index to angle

    # **Apply Weighting Factor to Pull Toward Center**
    adjusted_angle = (1 - lambda_weight) * angle  # Weighted angle calculation

    # Calculate dot position
    dot_x = int(np.interp(adjusted_angle, [-90, 90], [w, 0]))  # Map angle to x-position
    dot_y = h // 2  # Center vertically

    # Draw red dot
    cv2.circle(overlay, (dot_x, dot_y), radius=15, color=(0, 0, 255), thickness=-1)

    execution_time = time.time() - start_time
    print(f"Processed {image_path} in {execution_time:.4f} sec - Angle: {angle:.2f}° → Adjusted: {adjusted_angle:.2f}°")
    
    # Display results
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(yellow_mask, cmap='gray')
    plt.title("Yellow Mask")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(pink_mask, cmap='gray')
    plt.title("Pink Mask")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Processed + Heatmap + Direction ({adjusted_angle:.1f}°)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(detected_colors, cv2.COLOR_BGR2RGB))
    plt.title("Detected Colors (Random Contour Colors)")
    plt.axis("off")
    
    plt.show()
    plt.pause(0.3)

# Process images in folder
image_folder = "./image_folder2"
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

if not image_files:
    print("No images found in the folder!")

plt.ion()
for image_file in image_files:
    process_image(image_file, lambda_weight=0.5)  # Set λ (default = 0.5)
plt.ioff()
plt.show()