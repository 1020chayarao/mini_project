# -*- coding: utf-8 -*-
"""
image preprocessing
"""
#importing librabry
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# defining global variable path
image_path = "C:/Users/chaya/Desktop/mini"

'''function to load folder into arrays and 
then it returns that same array'''
def loadImages(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, 'digits', file)
         for file in os.listdir(path + "/digits") if      file.endswith('.jpeg')])
    return image_files


#second step:resizing

# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
 # Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# Preprocessing
def processing(data):
    # loading image
    # Getting 3 images to work with 
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]
    print('Original size',img[0].shape)
    # --------------------------------
    # setting dim of the resize
    height = 220
    width = 220
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
    # Checcking the size
    print("RESIZED", res_img[1].shape)
    # Visualizing one of the images in the array
    original = res_img[1]
    display_one(original)
 
    # ----------------------------------
    # Remove noise
    # Gaussian
    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        no_noise.append(blur)


    image = no_noise[1]
    display(original, image, 'Original', 'Blured')
   #---------------------------------
   # Segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Displaying segmented images
    display(original, thresh, 'Original', 'Segmented')
    
    # Further noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    #Displaying segmented back ground
    display(original, sure_bg, 'Original', 'Segmented Background')
    
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    
    # Displaying markers on the image
    display(image, markers, 'Original', 'Marked')
    
    #to grey scale image
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    display(original, gray, 'Original', 'grey scale')
    
    #gray scale to binary
    thresh = 127
    im_bw = cv2.threshold(original, thresh, 255, cv2.THRESH_BINARY)[1]
    display(original, im_bw, 'gray scale', 'binarised')
    
    
    
    
def main():
    # calling global variable
    global image_path
    '''The var Dataset is a list with all images in the folder '''          
    dataset = loadImages(image_path)
     
    print("List of files the first 3 in the folder:\n",dataset[:3])
    print("--------------------------------")
    
    # sending all the images to pre-processing
    pro = processing(dataset)
   
main()