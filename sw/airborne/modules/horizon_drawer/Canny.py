import cv2
import numpy as np
import glob
import math

#images = glob.glob( './flying_images/*.jpg', recursive=True)
images = glob.glob(r'C:\Users\stefh\Documents\low_to_the_ground_images\20250314-121408\*.jpg', recursive=True)

# ------------------ PARAMETERS ------------------
best_pixel_range = 30 #Amount of pixels in the width to calculate the averages for the best case
worst_pixel_range = 40 #Amount of pixels in the width to calculate the averages for the worst case
length_cutoff = 100 #The maximum length of lines to consider to reduce the noise
canny_low = 50 #The low threshold for the canny edge detection
canny_high = 130 #The high threshold for the canny edge detection
image_height_to_use = 180 #The height of the image to use for the edge detection
enable_blur = False #To apply a blur before the edge detection is performed
debug = False
# ------------------ END PARAMETERS ------------------


def convert_to_greyscale(img):
    """
    Converteer een afbeelding naar grijstinten zonder OpenCV-functies te gebruiken.
    """
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img[i][j] = 0.299 * img[i][j][2] + 0.587 * img[i][j][1] + 0.114 * img[i][j][0]
    return img


def blur(a):
    """
    Voer een 3x3 Gaussiaanse vervaging uit met een aangepaste kernel.
    """
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1) * kernel[y, x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum


def canny_self(img, low_treshold=canny_low, high_treshold=canny_high):
    """
    Eenvoudige implementatie van edge detection zonder OpenCV's Canny functie.
    """
    img_blank = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for row_pixel in range(0, img.shape[0] - 2 - 100):
        row_pixel = img.shape[0] - row_pixel - 2
        for col_pixel in range(0, img.shape[1] - 2):
            try:
                col_pixel = col_pixel + 1
                above = img[row_pixel + 1][col_pixel]
                below = img[row_pixel - 1][col_pixel]
                left = img[row_pixel][col_pixel - 1]
                right = img[row_pixel][col_pixel + 1]
                average = (abs(above[0]) + abs(below[0]) + abs(left[0]) + abs(right[0])) / 4
                if low_treshold < average < high_treshold:
                    img_blank = cv2.circle(img_blank, (col_pixel, row_pixel), 1, (255, 255, 255), 1)
            except Exception as exc:
                print(exc)
                continue

    return img_blank

def center_indices(lst):
    """
        Functie om de index van het midden te berekenen
    """
    length = len(lst)
    mid = length // 2
    return (mid - 1, mid) if length % 2 == 0 else (mid,)

if __name__ == "__main__":
    for f in images:
        img = cv2.imread(f)

        #Rotate the image 90 degrees counter clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Create a copy of the image
        img_copy = img.copy()

        #Apply a blur to the image
        if(enable_blur):
            img = blur(img)

        #Convert the image to gray scale
        img_self = convert_to_greyscale(img.copy())

        #Find the edges for the lower part of the image
        img_edges = canny_self(img_self)

        coordinate_list = []

        # Draw the lines from the bottom of the screen untill the first edge
        for i in range(0, img_edges.shape[1]):
            highestLength = 0

            for j in range(img_edges.shape[0] - 1, 0, -1):
                length = img_edges.shape[0] - j

                if(length > highestLength):
                    highestLength = length

                # If the pixel is white, draw a line from the bottom of the screen to the pixel
                if(length > length_cutoff):
                     continue

                if (img_edges[j][i] == 255):
                    cv2.line(img_copy, (i, img_edges.shape[0]), (i, j), (255, 0, 255), 1)
                    break

            if(highestLength > length_cutoff):
                continue

            coordinate_list.append((i, highestLength))

        X_vec = []
        X_vec_worst = []

        #Sort the list based on the length and get the width with the highest edges
        for x_index in range(0, math.ceil(len(coordinate_list)/best_pixel_range)):
            average = 0
            for i in range(0, best_pixel_range):
                if(x_index*best_pixel_range+i < len(coordinate_list)):
                    average += coordinate_list[x_index*best_pixel_range+i][1]
            average = average/best_pixel_range
            X_vec.append((average, x_index))

        #For the worst case
        for x_index in range(0, math.ceil(len(coordinate_list)/worst_pixel_range)):
            average = 0
            for i in range(0, worst_pixel_range):
                if(x_index*worst_pixel_range+i < len(coordinate_list)):
                    average += coordinate_list[x_index*worst_pixel_range+i][1]
            average = average/worst_pixel_range
            X_vec_worst.append((average, x_index))

        if(debug):
            # On the img_copy, draw the rectangles of width worst_pixel_range and show the average length of the edges
            for x in range(0, math.ceil(len(coordinate_list) / worst_pixel_range)):
                cv2.rectangle(img_copy, (x*worst_pixel_range, 0), (x*worst_pixel_range + worst_pixel_range, img_copy.shape[0]), (0, 0, 255), 2)
                cv2.putText(img_copy, str(int(X_vec_worst[x][0])), (x*worst_pixel_range, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate the performance for the center rectangle
        center_index = center_indices(X_vec)

        #Following variable returns the length of the center pixels(absolute value!)
        center_performance = (X_vec[center_index[0]][0] + X_vec[center_index[0]][0]) / 2 if len(center_index) == 2 else X_vec[center_index[0]][0]

        #Sort the x vector list based on the average
        X_vec.sort(key=lambda x: x[0])
        X_vec_worst.sort(key=lambda x: x[0])

        if(len(X_vec) == 0):
            print('continuing because of empty list')
            continue

        best_range = X_vec[len(X_vec) - 1][1]
        worst_range = X_vec_worst[0][1]

        #The best direction is [best_range * average_pixels ... best_range * average_pixels + average_pixels]
        #Create a rectangle on the image
        cv2.rectangle(img_copy, (best_range * best_pixel_range, 0), (best_range * best_pixel_range + best_pixel_range, img_copy.shape[0]), (0, 255, 0), 2)

        #Draw a rectangle over the best area
        cv2.rectangle(img_copy, (worst_range * worst_pixel_range, 0), (worst_range * worst_pixel_range + worst_pixel_range, img_copy.shape[0]), (255, 255, 255), 2)

        cv2.imshow('original', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()