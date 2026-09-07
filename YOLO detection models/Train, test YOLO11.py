# We are grateful for Ultralytics' work in this area of detection research! It is fantastic!
# Please see the tutorial here for more guidance on how to start: https://docs.ultralytics.com/quickstart/ and
# https://docs.ultralytics.com/models/yolov8/
# Also, see here for the Github repository for yolov5: https://github.com/ultralytics/ultralytics
#0806am

# All annotation data must be in YOLO format
# Your imagery and labels must be in a specific folder structure: /directory1/train/images and directory/train/labels AND
# /directory1/val/images and /directory1/val/labels ; These file paths will be specified in your opt.yaml file; please see template
# in this repository

# Once the Python requirements are met, you can specify vairables such as batch size, iou, epochs, imgsz (image size),
# patience, device, max_det (maximum detections), project and name where to save the results
import os
# from multiprocessing.spawn import freeze_support
import torch
import ultralytics

os.chdir("H:/WHCR_2025/2_detection/DATASET_detection_2025_v5/")
os.environ['KMP_DUPLICATE_LIB_OK']= "True"

# torch.backends.cudnn.enabled=True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# TRAIN A MODEL
# dataset download directory can be updated in 'C:\Users\aware\AppData\Roaming\Ultralytics\settings.json'

#time_start = time.time()

def main():
    model = ultralytics.YOLO("yolo11m.pt")
    model.info()

    # fastest -- cache = 'ram' , cache = 'disk', cache = False (slowest)
    results = model.train(data="H:/WHCR_2025/2_detection/new4.yaml",
                          batch= 16, #-1 to use suggestion
                          task="detect", epochs=70, # 100-300 epochs
                          imgsz=1024, patience=0, # typical patience 20-30
                          device= device, max_det=50, workers= 14,
                          lr0 = 0.0001,
                          cache= "disk", # can be false, disk, ram
                          optimizer= "Adam",
                          project="H:/WHCR_2025/2_detection/Aug2026/",
                          name = "yolo11m_aug19_lr0.50",
                          amp= True,
                          mosaic = 0.0,
                          conf=0.15)
if __name__ == "__main__":
    main()