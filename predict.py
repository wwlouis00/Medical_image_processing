from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


model = YOLO("your_model.pt")
model.info()

results = model.predict(source="your_image",project="your_save_dir",name="your_folder_name",save=True)

print(results)
