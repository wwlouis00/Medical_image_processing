from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


model = YOLO("ecg_detect.pt")
model.info()

results = model.predict(source="/root/Work/dataset/心電圖/img-00003-00001.jpg",project="/root/Work/yolov8/detect/",name="ecg",save=True)

print(results)
