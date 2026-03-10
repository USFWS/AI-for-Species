
from ultralytics import YOLO

model_path = "C:/users/aware/desktop/species_prep/model_weights/species_detector_yolo11s_Aug3.pt"
# Load the latest YOLO26 model
model = YOLO(model_path)

# Export the model to ONNX format with half-precision enabled
# The 'half=True' argument converts weights to FP16, reducing file size
# model.export(format= "onnx", half=True)

model.export(format= "onnx", half=True)

