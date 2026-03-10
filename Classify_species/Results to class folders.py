import os
import pandas
from os.path import basename
import shutil

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
root_path = "D:/species_2025/"
flight_name = "3_crops_all_inference"

csv_data = "D:/species_2025/3_crops_all_inference/classify_127k_rd2.csv"
# csv_data = root_path + flight_name + "/classify_" + flight_name + ".csv"
csv_data = pandas.read_csv(csv_data)
print(csv_data)

root_export = root_path + flight_name + "/classification_results/"
image_dir = root_path + flight_name + "/crops_birds_infer_127k/"
image_context = root_path + flight_name + "/bird_crops_w_context/"

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

