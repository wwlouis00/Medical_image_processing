from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

# from ultralytics.yolo.v8.

model = YOLO("chest_xray.pt")
model.info()
# model.predict("/root/Work/yolov8/box-2/box-2/valid/images/img_38_jpg.rf.a6d22ef06f98329c48da42429192c289.jpg",show=True,conf=0.5)

# Perform inference
# results = model.predict(source="/root/Work/dataset/Retinogram(R)/img-00001-00038.jpg",project="/root/Work/yolov8/object_detection/",name="retina(r)",save=True)
# results = model.predict(source="test2.png",save=True)
# print(results)
import glob
from IPython.display import Image, display

for image_path in glob.glob(f'/root/Work/dataset/chest/*jpg'):
      results = model.predict(source=image_path,project="/root/Work/yolov8/object_detection/",name="chest/chest",save=True)
      print("\n")