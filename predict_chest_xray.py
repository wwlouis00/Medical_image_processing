from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO("chest_xray.pt")
model.info()


import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/root/Work/dataset/chest/*jpg'):
      results = model.predict(source=image_path,project="/root/Work/yolov8/object_detection/",name="chest/chest",save=True)
      print("\n")
