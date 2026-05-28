import pandas as pd
import os
import shutil

# The script searches a image directory and copies/moves images that are listed in a csv file
# the csv file should have column labeled as "unique_image_jpg"

# Data inputs:
# csv data= list of images to be copied/moved (column header of 'unique_image_jpg')
# root_dir = directory of images to search
# dest1 = destination folder to move/copy images into

image_dir = "C:/users/bpickens/OneDrive - DOI/species_OneDrive/FINAL_species_PARENTS_377/"

csv_data = pd.read_csv("C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/2025_parents_add_to_train.csv")

dest1 =("C:/users/bpickens/OneDrive - DOI/species_OneDrive/detection/new_train_parents/")

if not os.path.exists(dest1):
    os.mkdir(dest1)

##if jpg is needed run this:
csv_data['unique_image_jpg'] = csv_data['unique_image_jpg'] #+ ".png"

x = 0

csv_list = []
csv_list = csv_data['unique_image_jpg'].tolist()
print(csv_list)

for root, dirs, files in os.walk(image_dir):
    for filename in files:
        if filename in csv_list:
            x = x + 1
            print ("Copied: ", x)
            path = os.path.join(root, filename)
            print ("path " , path)
            shutil.copy(path, dest1)  # can be shutil.move or shutil.copy
        else:
            pass

