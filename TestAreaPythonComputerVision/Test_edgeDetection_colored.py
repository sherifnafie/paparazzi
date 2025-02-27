#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:21:54 2025

@author: armin
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Ensure interactive mode is off for Spyder compatibility
plt.ioff()

# Measure start time
start_time = time.time()

# Load the image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
if image is None:
    raise FileNotFoundError("Image file not found")

# Convert image to YUV format
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Define color range in YUV format to detect all colored objects
lower_bound = np.array([0, 50, 50], dtype=np.uint8)  # Adjusted lower bound for colors
upper_bound = np.array([255, 160, 160], dtype=np.uint8)  # Adjusted upper bound for colors

# Create a mask for colored objects
mask = cv2.inRange(image_yuv, lower_bound, upper_bound)

# Use morphological operations to refine the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to extract colored objects
colored_objects = cv2.bitwise_and(image_yuv, image_yuv, mask=mask)

# Extract Y channel (luminance) for edge detection
y_channel = colored_objects[:, :, 0]

# Apply Gaussian blur to reduce noise
y_blurred = cv2.GaussianBlur(y_channel, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(y_blurred, 50, 150)

# Find contours from detected edges
min_object_size = 500  # Minimum contour area threshold
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# =============================================================================
# for contour in contours:
#     if cv2.contourArea(contour) > min_object_size:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image_yuv, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
# 
# =============================================================================

# Create a blank canvas to draw contours with different colors
contour_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for contour in contours:
    color = [random.randint(0, 255) for _ in range(3)]  # Generate a random color
    cv2.drawContours(contour_image, [contour], -1, color, 2)


# Measure end time
end_time = time.time()
execution_time = end_time - start_time

# Print execution time with color formatting
if execution_time > 1/30:
    print(f"\033[91mExecution Time: {execution_time:.4f} seconds\033[0m")  # Red text for slow execution
else:
    print(f"\033[92mExecution Time: {execution_time:.4f} seconds\033[0m")  # Green text for fast execution

# Display the results
plt.figure(figsize=(25, 5))
plt.subplot(1, 5, 1)
plt.title("Original Image with Bounding Boxes")
plt.imshow(cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB))

plt.subplot(1, 5, 2)
plt.title("Filtered Colored Objects")
plt.imshow(cv2.cvtColor(colored_objects, cv2.COLOR_YUV2RGB))

plt.subplot(1, 5, 3)
plt.title("Edge Detection using Canny")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 5, 4)
plt.title("Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(1, 5, 5)
plt.title("Contours with Random Colors")
plt.imshow(contour_image)



plt.show()