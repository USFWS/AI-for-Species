
from ultralytics import YOLO
import openvino
import onnxruntime as ort
import tensorrt
import config

#class UltralyticsDetectionModel(DetectionModel):
  #  """Detection model for Ultralytics YOLO models.
   # Supports PyTorch (.pt), ONNX (.onnx), OpenVINO (.xml or _openvino_model/),
    #NCNN (.param or _ncnn_model/), and TorchScript (.torchscript) models.
    #"""

model_path = config.MODEL_PATH_DETECT
# Load the latest YOLO26 model
model = YOLO(model_path)

# Export the model to ## format with half-precision enabled
# The 'half=True' argument converts weights to FP16
model.export(format= "onnx", half=True, imgz = 1024)
model.export(format= "engine", half=True)
# model.export(format= "openvino", half=True)



