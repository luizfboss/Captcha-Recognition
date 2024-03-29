import os
from torchvision.io import read_image
from torch.utils.data import Dataset

# a place to test some functionalities :)
file = open("one_digit_captcha.txt")
img_dir = 'one_digit_captcha/dev'


# def get_item(idx):
#         with open("one_digit_captcha.txt", 'r') as file:
#             line = file.readlines()[idx].split(',')
#             image_file = line[0] # image file name
#             img_path = os.path.join(img_dir, image_file)
#             image = read_image(img_path)
#             label = line[1].strip()
#         if transform:
#             image = transform(image)
#         if target_transform:
#             label = target_transform(label)
#         return image, label

# for i in range(0, 10):
#     print(get_item(i))

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = self.read_annotations_file(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    # format: [labels
    def read_annotations_file(self, annotations_file):
        with open(annotations_file, 'r') as file:
            lines = file.readlines()
        img_labels = [line.strip().split(',') for line in lines]
        return img_labels
    
# Assuming your text file is named 'annotations.txt'
annotations_file = 'four_digits_captcha_label.txt'
img_dir = 'four_digits_captcha/dev'

# Define your transform and target_transform if any
transform = None
target_transform = None

# Create an instance of CustomImageDataset
dataset = CustomImageDataset(annotations_file, img_dir, transform=transform, target_transform=target_transform)

# Test __len__ method
print("Length of dataset:", len(dataset))

# Test __getitem__ method
idx = 0
image, label = dataset[idx]
print("Image shape:", image.shape)
print("Label:", label)
    # def __getitem__(self, idx):
    #     img_filename, label = self.img_labels[idx]
    #     img_path = os.path.join(self.img_dir, img_filename)
    #     image = read_image(img_path)
    #     if self.transform:
    #         image = self.transform(image)
    #     if self.target_transform:
    #         label = self.target_transform(label)
    #     return image, label

