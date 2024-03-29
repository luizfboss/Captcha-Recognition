import os # used for path and image storage
from captcha.image import ImageCaptcha  # Module that will generate all captcha images# pip install captcha
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image # used for reading images in image storage
import random # sampling captcha text
import cv2
from PIL import Image


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # 36 classes
# 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
classes_len = len(classes) # 36 for now
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

# generating data for datasets - comment this piece of code to prevent the program from generating more images in case you already have the data.
# image_count = 80000 # number of images to be generated
# path = 'four_cap_36' # where the images will be stored.
# if not os.path.exists(path):
#     os.makedirs(path)
# for i in range(image_count):
#     text, image = gen_captcha_text_and_image()
#     filename = text+'-'+str(i)+'.png'
#     image_path = path  + '/' +  filename
#     image.save(image_path)
#     print(f'Saved {filename} in {path}')

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


four_cap_36_dataset = CustomImageDataset("four_cap_36.txt", "four_cap_36")

print(len(four_cap_36_dataset))
idx = 0
image, label = four_cap_36_dataset[idx]

# Convert the tensor to a NumPy array
image_np = image.numpy()


print("Image shape:", image_np)
print("Label:", label)