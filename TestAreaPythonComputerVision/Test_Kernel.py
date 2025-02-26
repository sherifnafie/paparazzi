# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:20:09 2025

@author: armin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
if image is None:
    raise FileNotFoundError("Image file not found")

# Convert to HSV format for color filtering
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for orange color
lower_orange = np.array([0, 100, 20])  # Adjust as needed
upper_orange = np.array([50, 255, 255])


# Create mask to filter only orange shades
mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
orange_image = cv2.bitwise_and(image, image, mask=mask)

# Convert to YUV format
yuv_image = cv2.cvtColor(orange_image, cv2.COLOR_BGR2YUV)

# Extract the Y channel (luminance)
y_channel = yuv_image[:, :, 0]

# Define Sobel kernel for edge detection in the x-direction
sobel_x = np.array([[-1, 0, 1], 
                     [-2, 0, 2], 
                     [-1, 0, 1]])

# Apply convolution using filter2D
edges_x = cv2.filter2D(y_channel, -1, sobel_x)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Filtered Orange Image")
plt.imshow(cv2.cvtColor(orange_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Edges in X Direction")
plt.imshow(edges_x, cmap='gray')

plt.show()

