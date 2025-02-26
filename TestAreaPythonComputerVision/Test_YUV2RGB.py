# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:48:19 2025

@author: armin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("image.jpg")  # Change this to your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display

# Convert to YUV color space
yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# Split Y, U, V channels
Y, U, V = cv2.split(yuv)

# Function to downsample and upsample chroma channels
def subsample_chroma(U, V, factor_x, factor_y):
    """ Applies chroma subsampling (downsampling + upsampling) """
    U_down = cv2.resize(U, (U.shape[1] // factor_x, U.shape[0] // factor_y), interpolation=cv2.INTER_AREA)
    V_down = cv2.resize(V, (V.shape[1] // factor_x, V.shape[0] // factor_y), interpolation=cv2.INTER_AREA)
    
    U_up = cv2.resize(U_down, (U.shape[1], U.shape[0]), interpolation=cv2.INTER_NEAREST)
    V_up = cv2.resize(V_down, (V.shape[1], V.shape[0]), interpolation=cv2.INTER_NEAREST)

    return U_up, V_up

# Apply different chroma subsampling methods
U_422, V_422 = subsample_chroma(U, V, 2, 1)  # 4:2:2 (Half horizontal resolution)
U_420, V_420 = subsample_chroma(U, V, 2, 2)  # 4:2:0 (Half horizontal & vertical resolution)
U_411, V_411 = subsample_chroma(U, V, 4, 1)  # 4:1:1 (Quarter horizontal resolution)

# Reconstruct YUV images
yuv_444 = cv2.merge([Y, U, V])         # Full chroma resolution (4:4:4)
yuv_422 = cv2.merge([Y, U_422, V_422]) # Half horizontal chroma (4:2:2)
yuv_420 = cv2.merge([Y, U_420, V_420]) # Half both horizontal & vertical chroma (4:2:0)
yuv_411 = cv2.merge([Y, U_411, V_411]) # Quarter horizontal chroma (4:1:1)

# Convert back to RGB for display
rgb_444 = cv2.cvtColor(yuv_444, cv2.COLOR_YUV2RGB)
rgb_422 = cv2.cvtColor(yuv_422, cv2.COLOR_YUV2RGB)
rgb_420 = cv2.cvtColor(yuv_420, cv2.COLOR_YUV2RGB)
rgb_411 = cv2.cvtColor(yuv_411, cv2.COLOR_YUV2RGB)

# Plot original and compressed images
fig, axs = plt.subplots(1, 5, figsize=(25, 5))

axs[0].imshow(image)
axs[0].set_title("Original (RGB)")

axs[1].imshow(rgb_444)
axs[1].set_title("YUV 4:4:4 (Full Chroma)")

axs[2].imshow(rgb_422)
axs[2].set_title("YUV 4:2:2 (Half Horiz. Chroma)")

axs[3].imshow(rgb_420)
axs[3].set_title("YUV 4:2:0 (Half Both Chroma)")

axs[4].imshow(rgb_411)
axs[4].set_title("YUV 4:1:1 (Quarter Horiz. Chroma)")

for ax in axs:
    ax.axis("off")

plt.show()

# Save results
cv2.imwrite("output_444.jpg", cv2.cvtColor(rgb_444, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_422.jpg", cv2.cvtColor(rgb_422, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_420.jpg", cv2.cvtColor(rgb_420, cv2.COLOR_RGB2BGR))
cv2.imwrite("output_411.jpg", cv2.cvtColor(rgb_411, cv2.COLOR_RGB2BGR))
