import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import dataset
import torch
import numpy as np
import PIL
import cv2


# def reduce_noise(channel):
#     # Apply noise reduction algorithm to the channel (e.g., Gaussian blur)
#     blurred = cv2.GaussianBlur(channel, (5, 5), 0)
# 
#     # You can apply additional noise reduction techniques as needed
#     # For example, thresholding, morphological operations, etc.
# 
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 
#     # Apply adaptive thresholding
#     _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 
#     # Apply morphological operations (closing) to further remove noise
#     kernel = np.ones((3, 3), np.uint8)
#     closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
#     return closed
# 
# def reduce_noise_captcha(image):
#     # Split the image into its RGB channels
#     channel_r, channel_g, channel_b = image[0], image[1], image[2]
# 
#     # Apply noise reduction to each channel
#     # channel_r_processed = reduce_noise(channel_r)
#     # channel_g_processed = reduce_noise(channel_g)
#     # channel_b_processed = reduce_noise(channel_b)
# 
#     # # Combine the processed channels back into a single image
#     processed_image = cv2.merge((channel_r, channel_g, channel_b))
# 
#     # return processed_image
#     return processed_image
# 

# four_cap_36_dataset = dataset.CustomImageDataset("four_cap_36.txt", "four_cap_36")

# print(len(four_cap_36_dataset))
# idx = 0
# image, label = four_cap_36_dataset[idx]
# print(image)
# print(label)
# image = image[0]

# Assuming 'image' is your tensor storing the image array with shape (height, width, channels)
# Convert the tensor to a numpy array

# Apply noise reduction to the image array
# reduced_image_array = reduce_noise_captcha(image_array)
# print(reduced_image_array)

# Convert the processed image array back to a tensor
# reduced_image = torch.from_numpy(reduced_image_array)

# Overwrite the tensor's image with the processed one
# image[:] = reduced_image[:]

# image = tensor_to_image(image)

import cv2
import easyocr
import matplotlib.pyplot as plt

# This needs to run only once to load the model into memory
reader = easyocr.Reader(['en'])

# reading the image
img = cv2.imread('four_cap_36/0A0I-31668.png')

# run OCR
results = reader.readtext(img)
print(results)

# show the image and plot the results
# plt.imshow(img)
# for res in results:
#     # bbox coordinates of the detected text
#     xy = res[0]
#     xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]
#     # text results and confidence of detection
#     det, conf = res[1], res[2]
#     # show time :)
#     plt.plot([xy1[0], xy2[0], xy3[0], xy4[0], xy1[0]], [xy1[1], xy2[1], xy3[1], xy4[1], xy1[1]], 'r-')
#     plt.text(xy1[0], xy1[1], f'{det} [{round(conf, 2)}]')
# 