import os
import pandas
from os.path import basename
import shutil
import config

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
root_path = config.EXPORT_DIR
csv_data = config.CSV_DATA

root_export = root_path + "/classification_results/"
image_dir = config.SOURCE_IMG
image_context = config.EXPORT_CONTEXT_BIRD

prob_threshold = 1.0

if not os.path.exists(root_export):
    os.mkdir(root_export)

dirs = os.listdir(image_dir)  # get all files in folder

for index, row in csv_data.iterrows():
    score1 = row['score1']
    print(score1)
    if score1 <  prob_threshold:
        source = image_context + row['unique_image_jpg']  # +'.jpg'
        print('Source : ', source)
        cat1 = row['label1']
        print("Class: ", cat1)

        for folders, subfolders, files in os.walk(image_context):
            name = basename(source)
            if name in files:
                dir2 = root_export + row['label1']
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                dest = root_export + row['label1'] + '/' + name
                print ("Destination : ", dest)
                shutil.copy(source, dest)  # this can be changed to: shutil.move
            else:
                pass
    else:
        print("Too high!")

