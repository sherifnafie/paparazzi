import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def detect_horizon_line(image_grayscaled):
    image_blurred = cv2.GaussianBlur(image_grayscaled, ksize=(3, 3), sigmaX=0)

    _, image_thresholded = cv2.threshold(image_blurred, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image_thresholded = image_thresholded - 1
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE, kernel=np.ones((9, 9), np.uint8))

    return image_closed

def rotate_and_crop(image, edges):
    # Detect horizon lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=75)
    if lines is None:
        return image  # Return original if no lines detected
    
    # Compute average angle of detected lines
    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in line]
    avg_angle = np.median(angles)
    
    # Rotate image to align horizon
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def process_image(image_path):
    start_time = time.time()
    image = cv2.imread(image_path)  
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Rotate image to correct orientation
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Convert to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Extract Y channel for edge detection
    y_channel_image = image_yuv[:,:,0]
    y_blurred_image = cv2.GaussianBlur(y_channel_image, (7, 7), 0)    
    overall_edges = cv2.Canny(y_blurred_image, 50, 150)
    
    # Rotate and align the horizon
    image = rotate_and_crop(image, overall_edges)
    
    # area detection from stef
    image_grayscale = image[:, :, 0]
    image_closed = (detect_horizon_line(image_grayscale))
    
    # Define color ranges for detection (modifiable values)
    color_ranges = {
        "orange": (np.array([5, 90, 90], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
        "yellow": (np.array([20, 100, 100], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
        "pink": (np.array([140, 40, 40], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8))
    }
    
    # Convert image to HSV for color segmentation
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for each color
    masks = {color: cv2.inRange(image_hsv, lower, upper) for color, (lower, upper) in color_ranges.items()}
    
    # Combine all masks
    combined_mask = sum(masks.values())
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # Apply mask to image to extract detected colors
    detected_colors = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Extract edges from the mask
    y_blurred = cv2.GaussianBlur(combined_mask, (7, 7), 0)
    edges = cv2.Canny(y_blurred, 50, 150)
    
    # Find contours in the edge-detected mask
    min_object_size = 400
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_object_size:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    execution_time = time.time() - start_time
    print(f"Processed {image_path} in {execution_time:.4f} seconds")
    
    # Display results
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(overall_edges, cmap='gray')
    plt.title("Overall Edges")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(image_closed, cmap='gray')
    plt.title("Mask")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Processed: {os.path.basename(image_path)}")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(detected_colors, cv2.COLOR_BGR2RGB))
    plt.title("Detected Colors")
    plt.axis("off")
    
    plt.show()
    plt.pause(0.1)

# Process images in folder
image_folder = "./image_folder2"
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
if not image_files:
    print("No images found in the folder!")
plt.ion()
for image_file in image_files:
    process_image(image_file)
plt.ioff()
plt.show()
