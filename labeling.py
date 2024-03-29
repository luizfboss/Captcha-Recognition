import os
 
# transfering data to a text file that stores image file name and its correspondent label.
# code source: https://www.geeksforgeeks.org/how-to-iterate-through-images-in-a-folder-python/
folder_dir = "four_cap_36" # change directory as needed
with open("four_cap_36.txt", 'a') as file: # change file name to the correct one
    for image in os.listdir(folder_dir):
        # check if the image ends with png
        if (image.endswith(".png")):
            image_name = image
            image_label = image.split("-")[0]
            file.write(f"{image_name},{image_label}\n")
            print(f"{image_name},{image_label}")