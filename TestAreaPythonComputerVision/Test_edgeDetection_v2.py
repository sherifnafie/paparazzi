# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:33:14 2025

@author: armin
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Measure start time
start_time = time.time()

# Load the image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
if image is None:
    raise FileNotFoundError("Image file not found")

# Convert image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define improved orange color range in RGB format
lower_orange = np.array([150, 60, 0], dtype=np.uint8)  # Expanded lower bound to capture darker shades
upper_orange = np.array([255, 180, 80], dtype=np.uint8)  # Adjusted upper bound

# Create a mask for orange colors
mask = cv2.inRange(image_rgb, lower_orange, upper_orange)

# Use morphological operations to refine the mask
# reduces noise and achieves more precise results - but works without
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to extract orange regions
orange_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Convert to grayscale for edge detection
gray = cv2.cvtColor(orange_image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur to reduce noise
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(gray_blurred, 50, 150)

# Find contours from detected edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box

# Measure end time
end_time = time.time()
execution_time = end_time - start_time

# Print execution time with color formatting
if execution_time > 1/30:
    print(f"\033[91mExecution Time: {execution_time:.4f} seconds\033[0m")  # Red text for slow execution
else:
    print(f"Execution Time: {execution_time:.4f} seconds")


# Display the results
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.title("Original Image with Bounding Boxes")
plt.imshow(image_rgb)

plt.subplot(1, 4, 2)
plt.title("Filtered Orange Image")
plt.imshow(orange_image)

plt.subplot(1, 4, 3)
plt.title("Edge Detection using Canny")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Mask")
plt.imshow(mask, cmap='gray')

plt.show()



# =============================================================================
# 
# 
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Load the image
# image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
# if image is None:
#     raise FileNotFoundError("Image file not found")
# 
# # Convert image to YUV format
# yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# 
# # Define orange color range in YUV format
# lower_orange = np.array([50, 100, 100], dtype=np.uint8)  # Adjusted values for YUV space
# upper_orange = np.array([255, 180, 130], dtype=np.uint8)
# # Define a refined orange color range in YUV format
# lower_orange = np.array([50, 110, 140], dtype=np.uint8)  # Adjusted values for better accuracy
# upper_orange = np.array([255, 150, 180], dtype=np.uint8)
# # Define a refined orange color range in YUV format
# lower_orange = np.array([30, 140, 100], dtype=np.uint8)  # Adjusted values for better accuracy
# upper_orange = np.array([200, 170, 130], dtype=np.uint8)
# lower_orange = np.array([50, 120, 120], dtype=np.uint8)  # Adjusted values for better accuracy
# upper_orange = np.array([255, 160, 140], dtype=np.uint8)
# 
# 
# # Create a mask based on color thresholding in YUV space
# mask = cv2.inRange(yuv_image, lower_orange, upper_orange)
# 
# # Use morphological operations to clean up the mask
# """ reduces noise """
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# 
# # Apply the mask to extract the orange regions
# detected_orange = cv2.bitwise_and(yuv_image, yuv_image, mask=mask)
# 
# # Extract the Y channel (luminance) for edge detection
# y_channel = detected_orange[:, :, 0]
# 
# # Apply Sobel filter in X direction and use absolute gradient
# edges_x = cv2.Sobel(y_channel, cv2.CV_64F, 1, 0, ksize=3)
# edges_x = np.abs(edges_x)  # Take absolute values for both edges
# detected_edges = np.uint8(edges_x)  # Convert back to 8-bit format
# 
# # Find contours from detected edges
# contours, _ = cv2.findContours(detected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(yuv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box in YUV image
# 
# # Convert YUV to RGB for visualization in Matplotlib
# image_rgb = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
# orange_rgb = cv2.cvtColor(detected_orange, cv2.COLOR_YUV2RGB)
# 
# # Display the results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("YUV Image with Bounding Boxes")
# plt.imshow(image_rgb)
# 
# plt.subplot(1, 3, 2)
# plt.title("Filtered Orange Image (YUV)")
# plt.imshow(orange_rgb)
# 
# plt.subplot(1, 3, 3)
# plt.title("Edges in X Direction")
# plt.imshow(detected_edges, cmap='gray')
# 
# plt.show()
# 
# 
# 
# """
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Load the image
# image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your file path
# if image is None:
#     raise FileNotFoundError("Image file not found")
# 
# # Convert image to RGB format
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 
# # Improve orange detection by including darker shades
# lower_orange = np.array([120, 60, 0], dtype=np.uint8)  # Expanded lower bound to include darker orange
# upper_orange = np.array([255, 200, 100], dtype=np.uint8)  # Upper bound remains the same
# 
# # Create a mask based on a more flexible color distance threshold
# mask = cv2.inRange(image_rgb, lower_orange, upper_orange)
# 
# # Use a morphological operation to clean up the mask
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# 
# orange_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
# 
# # Convert to YUV format
# yuv_image = cv2.cvtColor(orange_image, cv2.COLOR_RGB2YUV)
# 
# # Extract the Y channel (luminance)
# y_channel = yuv_image[:, :, 0]
# 
# # Apply Sobel filter in X direction and use absolute gradient to detect both edges
# edges_x = cv2.Sobel(y_channel, cv2.CV_64F, 1, 0, ksize=3)
# edges_x = np.abs(edges_x)  # Take absolute values to detect both edges
# edges_x = np.uint8(edges_x)  # Convert back to 8-bit format for display
# 
# # Find contours from edges to determine bounding boxes
# contours, _ = cv2.findContours(edges_x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
# 
# # Display the results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original Image with Bounding Boxes")
# plt.imshow(image_rgb)
# 
# plt.subplot(1, 3, 2)
# plt.title("Filtered Orange Image")
# plt.imshow(orange_image)
# 
# plt.subplot(1, 3, 3)
# plt.title("Edges in X Direction")
# plt.imshow(edges_x, cmap='gray')
# 
# plt.show()
# """
# =============================================================================
