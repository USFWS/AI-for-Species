import torch
import PIL
from PIL import Image
from torchvision import transforms
import csv
from os.path import basename
import os
import shutil
import pandas
import csv

# image_dir = directory of images to apply inference to
# root_export = directory where species folders are set up
# Optional (if new model is applied): idx_to_label = index to the corresponding label in the model
# model_path = pytorch classification model saved as script file
# transform_test = transform to be applied prior to inference

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
root_path = "D:/species_2025_2026/2026_species_thresh_0.906/"

# Input: subfolder of root_path above or flight name

flight_name = "inference_crops_birds"

root_export = root_path + flight_name + "/infer_species_score/"

new_csv = root_export + "/infer_species_threshold" + ".csv"

image_dir = root_path + flight_name

image_context = "C:/Users/aware/desktop/species_results/context_crops_birds/"
        # root_path + flight_name + "/species_infer_crops_test/")
#image_context = root_path + flight_name + "/crops_w_context_birds/"

model_path = "D:/species_2025/model_weights/species_classifier_Aug22_swin_s_rd3.pt"

prob_threshold = 1.00
max_label_index = 15
unlisted_object_index =12

if not os.path.exists(root_export):
    os.mkdir(root_export)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

# load model
model = torch.jit.load(model_path) #, map_location=torch.device(device))
model.to(device)

transform_test = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean= (0.2335, 0.2444, 0.2143), std=(0.1369,0.1149, 0.1031))
])

idx_to_label = {0: "Accipitridae", 1: "Anatidae", 2: "Ardeidae",3: "artificial", 4: "Charadriiformes",
                            5: "Laridae", 6: "Pelecanidae",7: "Phalacrocoracidae", 8: "Podicipedidae",
                            9: "Skimmer", 10: "Sterninae",
                            11: "Threskiornithidae", 12: "Unlisted_object",
                            13: "SACR", 14: "species", 15: "ROSP"
                            }

species_list = list(idx_to_label.values())

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', 'species_score', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6',
                     'score1', 'score2', 'score3', 'score4', 'score5', 'score6'])

def classify(model, transform_test, source):
    model = model.eval()
    image = PIL.Image.open(source)
    image = transform_test(image).float()
    image = image.to(device)
    image = image.unsqueeze(0)
    output = model(image)
    # print(output.data)
    softmax = torch.nn.functional.softmax(output, dim=1)

    top10_prob, top10_label = torch.topk(softmax, k = 10)

    label1 = top10_label[0, 0]
    label2 = top10_label[0, 1]
    label3 = top10_label[0, 2]
    label4 = top10_label[0, 3]
    label5 = top10_label[0, 4]
    label6 = top10_label[0, 5]
    label7 = top10_label[0, 6]
    label8 = top10_label[0, 7]
    label9 = top10_label[0, 8]
    label10 = top10_label[0, 9]

    score1 = top10_prob[0, 0]
    score2 = top10_prob[0, 1]
    score3 = top10_prob[0, 2]
    score4 = top10_prob[0, 3]
    score5 = top10_prob[0, 4]
    score6 = top10_prob[0, 5]
    score7 = top10_prob[0, 6]
    score8 = top10_prob[0, 7]
    score9 = top10_prob[0, 8]
    score10 = top10_prob[0, 9]

    if label1 > max_label_index:
        label1 = unlisted_object_index
    if label2 > max_label_index:
        label2 = unlisted_object_index
    if label3 > max_label_index:
        label3 = unlisted_object_index
    if label4 > max_label_index:
        label4 = unlisted_object_index
    if label5 > max_label_index:
        label5 = unlisted_object_index
    if label6 > max_label_index:
        label6 = unlisted_object_index
    if label7 > max_label_index:
        label7 = unlisted_object_index
    if label8 > max_label_index:
        label8 = unlisted_object_index
    if label9 > max_label_index:
        label9 = unlisted_object_index
    if label10 > max_label_index:
        label10 = unlisted_object_index

    #label1 = label1.data.cpu().numpy()
    #label2 = label2.data.cpu().numpy()
    #label3 = label3.data.cpu().numpy()
    #label4 = label4.data.cpu().numpy()
    #label5 = label5.data.cpu().numpy()
    #label6 = label6.data.cpu().numpy()
    #label7 = label7.data.cpu().numpy()
    #label8 = label8.data.cpu().numpy()
    #label9 = label9.data.cpu().numpy()
    #label10 = label10.data.cpu().numpy()

    score1 = score1.data.cpu().numpy()
    score2 = score2.data.cpu().numpy()
    score3 = score3.data.cpu().numpy()
    score4 = score4.data.cpu().numpy()
    score5 = score5.data.cpu().numpy()
    score6 = score6.data.cpu().numpy()
    score7 = score7.data.cpu().numpy()
    score8 = score8.data.cpu().numpy()
    score9 = score9.data.cpu().numpy()
    score10 = score10.data.cpu().numpy()

    print("Label10: ", label10)

    species_list = list(idx_to_label.values())

    label1 = species_list[label1]
    label2 = species_list[label2]
    label3 = species_list[label3]
    label4 = species_list[label4]
    label5 = species_list[label5]
    label6 = species_list[label6]
    label7 = species_list[label7]
    label8 = species_list[label8]
    label9 = species_list[label9]
    label10 = species_list[label10]

    print(label1, label2, label3)
    species_score = 0
    ## Single out species data only
    if label1 == "species":
        species_score = score1
        print("species!!!!!!")
    if label2 == "species":
        species_score = score2
    if label3 == "species":
        species_score = score3
    if label4 == "species":
        species_score = score4
    if label5 == "species":
        species_score = score5
    if label6 == "species":
        species_score = score6
    if label7 == "species":
        species_score = score7
    if label8 == "species":
        species_score = score8
    if label9 == "species":
        species_score = score9
    if label10 == "species":
        species_score = score10


    with open(new_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, species_score, label1, label2, label3, label4, label5, label6, score1, score2, score3, score4,
                         score5, score6])
x = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(".jpg"):
            source = os.path.join(root, file)
            name = os.path.basename(source)
            print ("name: ", name)
            classify(model, transform_test, source)
            x = x+1
            print(x, "are classified")
        else:
            pass
###############
# This part does the moving
dirs = os.listdir(image_dir)  # get all files in folder

csv_data = pandas.read_csv(new_csv)

for index, row in csv_data.iterrows():
    score1 = row['score1']
    score2 = row['score2']
    score3 = row['score3']
    label1 = row['label1']
    label2 = row['label2']
    label3 = row['label3']

    if score1 <  prob_threshold:
        target = image_context + "/" + row['unique_image_jpg']  # +'.jpg'
        print('Target : ', target)
        cat1 = row['label1']
        print("Class: ", cat1)

        for folders, subfolders, files in os.walk(image_context):
            name = basename(target)
            print ("name: ", name)
            if name in files:
                dir2 = root_export + "/" + row['label1']
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                dest = root_export + "/" + row['label1'] + '/' + name

                print ("Destination : ", dest)
                shutil.copy(target, dest)  # this can be changed to: shutil.move
            else:
                pass

            if label2 == "species": # and score2 > 0.00:
                print("label2- whooper!")
                print("name: ", name)
                if name in files:
                    dir3 = root_export + "/" + row['label1'] + "_" + row['label2']
                    if not os.path.exists(dir3):
                        os.makedirs(dir3)
                    dest = root_export + "/" + row['label1'] + "_" + row['label2'] + "/" + name

                    print("Destination species2: ", dest)
                    shutil.copy(target, dest)  # this can be changed to: shutil.move
                else:
                    pass
