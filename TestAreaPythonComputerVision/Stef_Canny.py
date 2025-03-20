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
images = glob.glob(r'/home/armin/Documents/MAV/TestAreaPythonComputerVision/image_folder2/*.jpg', recursive=True)

average_pixels = 20
worst_pixel_range = 100
length_cutoff = 100

for f in images:
    img = cv2.imread(f)

    #Rotate the image 90 degrees counter clockwise
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Create a copy of the image
    img_copy = img.copy()

    #Apply a guassian blur to the image
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Apply canny edge
    edges = cv2.Canny(img, 50, 130)

    # show the edges image
    #cv2.imshow('edges', edges)

    coordinate_list = []

    # Draw the lines from the bottom of the screen untill the first edge
    for i in range(0, edges.shape[1]):
        highestLength = 0

        for j in range(edges.shape[0] - 1, 0, -1):
            length = edges.shape[0] - j

            if(length > highestLength):
                highestLength = length

            #coordinate_list.append((i, j, length))

            # If the pixel is white, draw a line from the bottom of the screen to the pixel
            if(length > length_cutoff):
                 continue

            if (edges[j][i] == 255):
                cv2.line(img_copy, (i, edges.shape[0]), (i, j), (255, 0, 255), 1)
                break

        if(highestLength > length_cutoff):
            continue

        coordinate_list.append((i, highestLength))
    #print(coordinate_list)

    X_vec = []
    X_vec_worst = []
    for x_index in range(0, math.ceil(len(coordinate_list)/average_pixels)):
        average = 0
        for i in range(0, average_pixels):
            if(x_index*average_pixels+i < len(coordinate_list)):
                average += coordinate_list[x_index*average_pixels+i][1]
        average = average/average_pixels
        X_vec.append((average, x_index))
        #y_vec.append(1)

    #For the worst case
    for x_index in range(0, math.ceil(len(coordinate_list)/worst_pixel_range)):
        average = 0
        for i in range(0, average_pixels):
            if(x_index*worst_pixel_range+i < len(coordinate_list)):
                average += coordinate_list[x_index*worst_pixel_range+i][1]
        average = average/worst_pixel_range
        X_vec_worst.append((average, x_index))

    #Sort the x vector list based on the average
    X_vec.sort(key=lambda x: x[0])
    X_vec_worst.sort(key=lambda x: x[0])

    print(X_vec)

    if(len(X_vec) == 0):
        continue

    best_range = X_vec[len(X_vec) - 1][1]
    worst_range = X_vec_worst[0][1]

    # Draw rectangles
    cv2.rectangle(img_copy, (best_range * average_pixels, 0), (best_range * average_pixels + average_pixels, img_copy.shape[0]), (0, 255, 0), 2)
    cv2.rectangle(img_copy, (worst_range * average_pixels, 0), (worst_range * average_pixels + average_pixels, img_copy.shape[0]), (0, 0, 255), 2)
    
    # Display using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Processed: {f}")
    plt.axis("off")
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
    
