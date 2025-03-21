import numpy as np
import cv2
import matplotlib.pyplot as plt

# Extract the Y (luminance) channel from YUV422 image buffer
def extract_y(img_yuv422, width, height):
    gray = np.zeros((height, width), dtype=np.uint8)
    
    for py in range(height):
        row_start = py * width * 2
        for px in range(width):
            if px % 2 == 0:
                gray[py, px] = img_yuv422[row_start + 2 * px + 1]
            else:
                gray[py, px] = img_yuv422[row_start + 2 * px - 1]
    return gray

# Apply Sobel edge detection with thresholding
def sobel_edge(gray_img, edge_thresh=80):
    h, w = gray_img.shape
    edge_out = np.zeros_like(gray_img, dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v00 = int(gray_img[y-1, x-1])
            v01 = int(gray_img[y-1, x])
            v02 = int(gray_img[y-1, x+1])
            v10 = int(gray_img[y, x-1])
            v12 = int(gray_img[y, x+1])
            v20 = int(gray_img[y+1, x-1])
            v21 = int(gray_img[y+1, x])
            v22 = int(gray_img[y+1, x+1])

            gx = (-v00 + v02 - 2*v10 + 2*v12 - v20 + v22)
            gy = (v00 + 2*v01 + v02 - v20 - 2*v21 - v22)
            mag = abs(gx) + abs(gy)

            if mag > edge_thresh:
                edge_out[y, x] = 255
    return edge_out

# Find the column with the greatest distance to first edge pixel from bottom
def find_best_column(edge_img):
    h, w = edge_img.shape
    best_col = -1
    best_val = -1

    for x in range(w):
        dist = 0
        found_edge = False
        for y in range(h-1, -1, -1):
            if edge_img[y, x] == 255:
                dist = (h - 1) - y
                found_edge = True
                break
        if not found_edge:
            dist = h
        if dist > best_val:
            best_val = dist
            best_col = x
    return best_col, best_val

# Main processing function
def horizon_drawer_detect(img_yuv422, width, height):
    gray = extract_y(img_yuv422, width, height)
    edges = sobel_edge(gray)

    best_col, best_dist = find_best_column(edges)

    min_safe_dist = height // 4
    if best_dist < min_safe_dist:
        black_percent = 0.0
    else:
        black_percent = 100.0

    print(f"Canny-like best_col={best_col} best_dist={best_dist} => black%={black_percent:.1f}")
    return edges, best_col, black_percent



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

import cv2
import numpy as np
import glob
#from sklearn.model_selection import train_test_split
#from sklearn.tree import export_text
from random import randrange
import time
import math

#images = glob.glob( './flying_images/*.jpg', recursive=True)
#images = glob.glob(r'C:\Users\stefh\Documents\low_to_the_ground_images\20250314-121408\*.jpg', recursive=True)
#images = glob.glob(r'\home\armin\Documents\MAV\TestAreaPythonComputerVision\*.jpg', recursive=True)
images = glob.glob(r'/home/armin/Documents/MAV/TestAreaPythonComputerVision/image_folder4/*.jpg', recursive=True)

average_pixels = 20
worst_pixel_range = 100
length_cutoff = 100

for f in images:
    img = cv2.imread(f)

    #Rotate the image 90 degrees counter clockwise
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


    if img is None:
        print(f"Failed to load image")
    else:
        print("Image loaded successfully.")

        # Convert BGR to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


    width, height = 640, 480
    edges, best_col, black_percent = horizon_drawer_detect(img_yuv, width, height)

    plt.imshow(edges, cmap='gray')
    plt.title("Edge Map")
    plt.show()

    

# =============================================================================
#     #The best direction is [best_range * average_pixels ... best_range * average_pixels + average_pixels]
#     #Create a rectangle on the image
#     cv2.rectangle(img_copy, (best_range * average_pixels, 0), (best_range * average_pixels + average_pixels, img_copy.shape[0]), (0, 255, 0), 2)
# 
#     cv2.rectangle(img_copy, (worst_range * average_pixels, 0), (worst_range * average_pixels + average_pixels, img_copy.shape[0]), (0, 0, 255), 2)
# 
#     cv2.imshow('original', img_copy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# =============================================================================
    