import os # used for path and image storage
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont # used for reading and generating images
import random # sampling captcha text
import dataset

no_noise_size =  4 # how many characters in the image

def generate_random_text():
    text = ""
    for i in range(no_noise_size):
        character = random.choice(dataset.classes)
        text += character
    return text

def create_image_no_noise():
    message = generate_random_text()
    font = ImageFont.truetype("arial.ttf", size=20)
    img = Image.new('RGB', (dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT), color='white')
    imgDraw = ImageDraw.Draw(img)

    # for images with more than one character:
    # if we have more than one character in the text, add some spacing in the image. 
    if len(message) > 1:
        image_text = ""
        for character in message:
            image_text += character + "  "      
        imgDraw.text((20, 20), image_text, font=font, fill=(0, 0, 0))
    
    # for images with less only one character:
    # range of text position in the image: 20-120 (for images with 1 char)
    else:
        char_position = random.randint(20, 120)
        imgDraw.text((char_position, 20), message, font=font, fill=(0, 0, 0))
    return message, img


# generating data for datasets
image_count = 12000 # number of images to be generated
path = dataset.test_dataset_path # where the images will be stored. this can be any of the paths above (TRAIN_DATASET_PATH, TEST_DATASET_PATH, DEV_DATASET_PATH)
# # Note: I believe that this is an easier approach to data splitting because personally, it is easier to visualize how the data preparation and splitting works. I could use pytorch to do that but this way makes it easier for me to understand.
if not os.path.exists(path):
    os.makedirs(path)
for i in range(image_count):
    text, image = create_image_no_noise()
    filename = text+'-'+str(i)+'.png' # adding an "id" (i) to the image will prevent images from being overwritten when data is being generated. For example, if an image named "a.png" has already been generated, if I don't add an ID in the image, another image would be generated with the same name, and this would delete the information from the first one and rewrite it with the information of the second one.
    image_path = path  + '/' +  filename
    image.save(image_path)
    print(f'Saved {filename} in {path}')
