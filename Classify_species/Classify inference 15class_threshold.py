import torch
import PIL
from PIL import Image
from torchvision import transforms
from os.path import basename
import os
import shutil
import pandas
import csv
import config

# image_dir = directory of images to apply inference to
# root_export = directory where species folders are set up
# Optional (if new model is applied): idx_to_label = index to the corresponding label in the model
# model_path = pytorch classification model saved as script file
# imagee_context = folder with crops with context

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
image_dir = config.SOURCE_IMG
root_export = config.EXPORT_DIR

model_path = config.MODEL_PATH_CLASSIFY
new_csv = root_export + "/infer_2025_" + "Aug27_results.csv"

image_context = config.EXPORT_CONTEXT_BIRD

prob_threshold = 1.00

if not os.path.exists(root_export):
    os.mkdir(root_export)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

# load model
model = torch.jit.load(model_path)
model.to(device)

transform_test = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean= (0.2335, 0.2444, 0.2143), std=(0.1369,0.1149, 0.1031))
])

idx_to_label = {0: "Accipitridae", 1: "Anatidae", 2: "Ardeidae",
                            3: "artificial", 4: "Charadriiformes",
                            5: "Laridae", 6: "Pelecanidae",
                            7: "Phalacrocoracidae", 8: "Podicipedidae",
                            9: "Skimmer",
                            10: "Sterninae",
                            11: "Threskiornithidae", 12: "Unlisted_object",
                            13: "SACR", 14: "WHCR", 15: "ROSP"
                            }

species_list = list(idx_to_label.values())

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['unique_image_jpg', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'score1', 'score2', 'score3',
         'score4', 'score5', 'score6', 'species_prob'])

def classify(model, transform_test, source):
    model = model.eval()
    image = PIL.Image.open(source)
    image = transform_test(image).float()
    image = image.to(device)
    image = image.unsqueeze(0)
    image = image.half()
    output = model(image)
    # print(output.data)
    softmax = torch.nn.functional.softmax(output, dim=1)

    top6_prob, top6_label = torch.topk(softmax, 6)
    # print("tops: ", top3_prob,top3_label)
    label1 = top6_label[0, 0]
    label2 = top6_label[0, 1]
    label3 = top6_label[0,2]
    label4= top6_label[0, 3]
    label5 = top6_label[0, 4]
    label6 = top6_label[0, 5]

    score1 = top6_prob[0, 0]
    score2 = top6_prob[0, 1]
    score3 = top6_prob[0,2]
    score4 = top6_prob[0, 3]
    score5 = top6_prob[0, 4]
    score6 = top6_prob[0, 5]

    label1 = label1.data.cpu().numpy()
    label2 = label2.data.cpu().numpy()
    label3 = label3.data.cpu().numpy()
    label4 = label4.data.cpu().numpy()
    label5 = label5.data.cpu().numpy()
    label6 = label6.data.cpu().numpy()

    score1 = score1.data.cpu().numpy()
    score2 = score2.data.cpu().numpy()
    score3 = score3.data.cpu().numpy()
    score4 = score4.data.cpu().numpy()
    score5 = score5.data.cpu().numpy()
    score6 = score6.data.cpu().numpy()
    species_list = list(idx_to_label.values())

    label1 = species_list[label1]
    label2 = species_list[label2]
    label3 = species_list[label3]
    label4 = species_list[label4]
    label5 = species_list[label5]
    label6 = species_list[label6]

    print(label1, label2, label3)
    species_prob = 0.00
    if label1 == "WHCR":
        species_prob = score1
        print("WHCR!!!!!!")
    if label2 == "WHCR":
        species_prob = score2
    if label3 == "WHCR":
        species_prob = score3
    if label4 == "WHCR":
        species_prob = score4
    if label5 == "WHCR":
        species_prob = score5
    if label6 == "WHCR":
        species_prob = score6

    with open(new_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, label1, label2, label3, label4, label5, label6, score1, score2, score3, score4, score5, score6, species_prob])
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
    label1 = row['label1']
    label2 = row['label2']
    label3 = row['label3']
    label4 = row['label4']
    label5 = row['label5']
    label6 = row['label6']

    score1 = row['score1']
    score2 = row['score2']
    score3 = row['score3']
    score4 = row['score4']
    score5 = row['score5']
    score6 = row['score6']

    target = image_context + "/" + row['unique_image_jpg']  # +'.jpg'
    print('Target : ', target)
    cat1 = row['label1']
    print("Class: ", cat1)

    for folders, subfolders, files in os.walk(image_context):
        name = basename(target)
        print("name: ", name)
        if name in files:
            dir2 = root_export + "/" + row['label1']
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            dest = root_export + "/" + row['label1'] + '/' + name

            print("Destination : ", dest)
            shutil.copy(target, dest)  # this can be changed to: shutil.move
        else:
            pass

        if label2 == "WHCR":
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
