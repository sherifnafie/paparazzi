#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:57:07 2025

@author: armin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def process_image(image_path):
    # Measure start time
    start_time = time.time()

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return None
    
    # Rotate image 90 degrees clockwise
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define improved orange color range in RGB format
    lower_orange = np.array([150, 60, 0], dtype=np.uint8)  # Expanded lower bound to capture darker shades
    upper_orange = np.array([255, 180, 80], dtype=np.uint8)  # Adjusted upper bound

    # Create a mask for orange colors
    mask = cv2.inRange(image_rgb, lower_orange, upper_orange)

    # Use morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to extract orange regions
    orange_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert to grayscale for edge detection
    if np.any(orange_image):  # Check if orange_image is not completely black
        gray = cv2.cvtColor(orange_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.zeros_like(mask)  # Avoid errors if no orange pixels are detected

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
    plt.figure(figsize=(10, 5))
    plt.imshow(image_rgb)
    plt.title(f"Processed Image: {os.path.basename(image_path)}")
    plt.pause(1/30)
    plt.clf()

# Specify folder containing images
image_folder = "./image_folder"  # Replace with your folder path
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

# Process each image in the folder
plt.ion()  # Turn on interactive mode for live updating
for image_file in image_files:
    process_image(image_file)

plt.ioff()  # Turn off interactive mode
plt.show()
