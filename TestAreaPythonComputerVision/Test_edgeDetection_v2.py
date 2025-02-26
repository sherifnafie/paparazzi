# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:33:14 2025

@author: armin
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
if image is None:
    raise FileNotFoundError("Image file not found")

# Convert image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Improve orange detection by including darker shades
lower_orange = np.array([120, 60, 0], dtype=np.uint8)  # Expanded lower bound to include darker orange
upper_orange = np.array([255, 200, 100], dtype=np.uint8)  # Upper bound remains the same

# Create a mask based on a more flexible color distance threshold
mask = cv2.inRange(image_rgb, lower_orange, upper_orange)

# Use a morphological operation to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

orange_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Convert to YUV format
yuv_image = cv2.cvtColor(orange_image, cv2.COLOR_RGB2YUV)

# Extract the Y channel (luminance)
y_channel = yuv_image[:, :, 0]

# Apply Sobel filter in X direction and use absolute gradient to detect both edges
edges_x = cv2.Sobel(y_channel, cv2.CV_64F, 1, 0, ksize=3)
edges_x = np.abs(edges_x)  # Take absolute values to detect both edges
edges_x = np.uint8(edges_x)  # Convert back to 8-bit format for display

# Find contours from edges to determine bounding boxes
contours, _ = cv2.findContours(edges_x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image with Bounding Boxes")
plt.imshow(image_rgb)

plt.subplot(1, 3, 2)
plt.title("Filtered Orange Image")
plt.imshow(orange_image)

plt.subplot(1, 3, 3)
plt.title("Edges in X Direction")
plt.imshow(edges_x, cmap='gray')

plt.show()