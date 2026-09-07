import torch
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import DataLoader
import os
import PIL
import pandas
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
import time

import config

model_path = config.MODEL_PATH_CLASSIFY

root_path = "H:/WHCR_2025/1_classify/"
model_name = root_path + "results_classify_v6/"

# Inputs
csv_test = "H:/WHCR_2025/1_classify/DATASET_classify_v6/classify_test_v6.csv"
image_folder = "H:/WHCR_2025/1_classify/crops_small_2025/"

# These metrics are automatically reported; no inputs needed
classification_report = model_name + "_prediction_report.csv"
raw_confusion_matrix = model_name + "_raw_conf_matrix.csv"
normalized_confusion_matrix = model_name+ "_normal_conf_matrix.csv"
confusion_matrix_png = model_name + "_normal_conf_matrix_pic.png"
========

model_path = "D:/species_2025/model_weights/species_classifier_Aug22_swin_s_rd3.pt"

model_name = "species_rd4_test_dataset"

# Inputs
csv_test = "D:/species_2025/6_classify/DATASETS/rd4_dataset/test_dataset.csv"
image_folder = "D:/species_2025/6_classify/DATASETS/rd4_dataset/test_dataset_crops/"

classification_report = "D:/species_2025/6_classify/DATASETS/rd4_dataset/" + model_name + "_prediction_report .csv"
raw_confusion_matrix = "D:/species_2025/6_classify/DATASETS/rd4_dataset/" + model_name + "_raw_conf_matrix.csv"
normalized_confusion_matrix = "D:/species_2025/6_classify/DATASETS/rd4_dataset/" + model_name + "_normal_conf_matrix.csv"
confusion_matrix_png = "D:/species_2025/6_classify/DATASETS/rd4_dataset/" + model_name + "_normal_conf_matrix_pic.png"
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py

###########
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
if not os.path.exists(model_name):
    os.mkdir(model_name)

start_time = time.time()

========
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py
class2index = {"Accipitridae": 0, "Anatidae": 1, "Ardeidae": 2,
                            "artificial": 3, "Charadriiformes": 4,
                            "Laridae": 5, "Pelecanidae": 6,
                            "Phalacrocoracidae": 7, "Podicipedidae": 8,
                            "Skimmer": 9,
                            "Sterninae": 10,
                            "Threskiornithidae": 11, "Unlisted_object": 12,
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
                            "SACR": 13, "WHCR": 14, "ROSP": 15
========
                            "SACR": 13, "species_adult": 14, "ROSP": 15, "species_juvenile": 16
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py
                            }
# load model
model = torch.jit.load(model_path)
model.to(device)

# test dataset
class CustomDataset(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_path, image_folder, transform):
        self.annotations = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform_test
        self.class2index = {"Accipitridae": 0, "Anatidae": 1, "Ardeidae": 2,
                            "artificial": 3, "Charadriiformes": 4,
                            "Laridae": 5, "Pelecanidae": 6,
                            "Phalacrocoracidae": 7, "Podicipedidae": 8,
                            "Skimmer": 9,
                            "Sterninae": 10,
                            "Threskiornithidae": 11, "Unlisted_object": 12,
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
                            "SACR": 13, "WHCR_adult": 14, "ROSP": 15
========
                            "SACR": 13, "species_adult": 14, "ROSP": 15, "species_juvenile": 16
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py
                            }
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.annotations.iloc[index, 1])  # r, c; col is image name
        image = PIL.Image.open(img_path)
        image = transform_test(image)
        label = self.annotations.iloc[index, 0]
        label = torch.tensor(label)  #################formerly torch.tensor
        return (image, label)  # records the item


transform_test = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=(0.2335, 0.2444, 0.2143), std=(0.1369, 0.1149, 0.1031))
])

test_dataset = CustomDataset(csv_path=csv_test, image_folder=image_folder, transform=transform_test)

y = len(test_dataset)

print("test dataset: ", y)

## Get test predictions
test_loader2 = DataLoader(test_dataset, shuffle=False)

test_pred_list = []
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
x=0
========
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py

with torch.no_grad():
    model.eval()
    for images, labels in test_loader2:
        images, labels = images.to(device), labels.to(device)
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
        images = images.half()
========
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py
        test_pred = model.forward(images)

        _, test_pred_classes = torch.max(test_pred, dim=1)

        probs = torch.softmax(test_pred, dim=1)
        preds = probs.argmax(dim=1)
        test_pred_list.append(test_pred_classes.cpu().numpy())
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
        x = x + 1
        print("Classified: ", x)
========
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py

# below are class index predictions
test_pred_list = [a.squeeze().tolist() for a in test_pred_list]

## Classification report
label_truth = []
for images, labels in test_loader2:
    label_truth.append(labels.cpu().numpy())

score = accuracy_score(label_truth, test_pred_list)
bal_score = balanced_accuracy_score(label_truth, test_pred_list)
print("Overal accuracy is: ", score)
print("Balanced accuracy is: ", bal_score)

<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
print("Start time: ", start_time)
end_time = time.time()
duration_min = (end_time - start_time)/60
duration_sec = duration_min*60
duration_hrs = duration_min / 60

print("duration (seconds): ", duration_sec)
print("duration (minutes): ", duration_min)
print("duration (hours): ",duration_hrs)

========
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py
# label_truth = [a.squeeze().tolist() for a in label_list]

# Read in class indices
class_names = list(class2index.keys())
class_index = list(class2index.values())

## Recall, precision stats report
report1 = sklearn.metrics.classification_report(label_truth, test_pred_list, output_dict=True, labels=class_index,
                                                target_names=class_names,
                                                zero_division=False)
classify_report = pandas.DataFrame(report1).transpose()
classify_report.to_csv(classification_report)

# raw numbers for confusion matrix
class_list = list(class2index.keys())
#class_index = list(test_pred_list)

cm = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index)
cm = pandas.DataFrame(cm).transpose()
class_list = list(class2index.keys())
cm.to_csv(raw_confusion_matrix, header= class_list)

# Normalized confusion matrix
<<<<<<<< HEAD:Classify- train, test/Classify train, test _15class- part2_half.py
cm_normalized = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index, normalize = "true")
cm_normalized = pandas.DataFrame(cm_normalized).transpose()
class_list = list(class2index.keys())
cm_normalized.to_csv(normalized_confusion_matrix, header= class_list)
========
report1 = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index, normalize = "true")
df = pandas.DataFrame(report1).transpose()
class_list = list(class2index.keys())
cm.to_csv(normalized_confusion_matrix, header= class_list)
>>>>>>>> origin/main:Classify- train, test/Classify train, test - part2.py

# Plot confusion matrix
label_list = []
for images, labels in test_loader2:
    label_list.append(labels.cpu().numpy())

label_list = [a.squeeze().tolist() for a in label_list]

idx2class = {v: k for k, v in class2index.items()}

confusion_matrix_df = pd.DataFrame(confusion_matrix(label_list, test_pred_list)).rename(columns=idx2class,
                                                                                        index=idx2class)

# sns.set(rc = {'figure.figsize':(16,8)})
plt.rcParams['figure.dpi'] = 200
plt.figure(figsize=(26, 20))
# plt.rcParams['savefig.dpi'] = 300

final1 = sns.heatmap(confusion_matrix_df / numpy.sum(confusion_matrix_df), annot=True,
                     fmt='.0%', cmap='Blues')

final1.figure.savefig(confusion_matrix_png)