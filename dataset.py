import os # used for path and image storage
from captcha.image import ImageCaptcha  # Module that will generate all captcha images# pip install captcha
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image # used for reading images in image storage
import random # sampling captcha text
import cv2

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
classes_len = len(classes)
captcha_size = 4

image_height = 60
image_width = 160


# Generate random captcha text for image
def random_captcha_text():
    captcha_text = ""
    for c in range(captcha_size):
        character = random.choice(classes)
        captcha_text += character
    return captcha_text

# generate image with the text generated from random_captcha_text() function
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

# generating data for datasets
image_count = 80000 # number of images to be generated
path = 'four_dig_cap' # where the images will be stored.
if not os.path.exists(path):
    os.makedirs(path)
for i in range(image_count):
    text, image = gen_captcha_text_and_image()
    filename = text+'-'+str(i)+'.png'
    image_path = path  + '/' +  filename
    image.save(image_path)
    print(f'Saved {filename} in {path}')

# Creating a custom dataset. 
# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = self.read_annotations_file(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path) # reads image from image path
        
        image = torch.Tensor(image)
        return image, label

    # returns an array wuth many sub arrays that contain [image_file_name.png, label]
    def read_annotations_file(self, annotations_file):
        with open(annotations_file, 'r') as file:
            lines = file.readlines()
        img_labels = [line.strip().split(',') for line in lines]
        return img_labels

# one_dig_no_noise_dev = CustomImageDataset('one_dig_no_dev.txt', "one_digit_no_noise/dev")
# one_dig_no_noise_test = CustomImageDataset('one_dig_no_test.txt', "one_digit_no_noise/test")

# print(len(one_dig_no_noise_dev))
# idx = 0
# image, label = one_dig_no_noise_dev[idx]
# print("Image shape:", image)
# print("Label:", label)