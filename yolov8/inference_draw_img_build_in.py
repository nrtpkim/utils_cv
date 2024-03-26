import pandas as pd
from ultralytics import YOLO
import os

### Get data img path
image_paths = []
MAIN_DATA_FOLDER = 'test/round 2'
for root, dirs, files in os.walk(MAIN_DATA_FOLDER):
    for file in files:
        
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".avif")):
            image_path = os.path.join(root, file)
            image_paths.append(image_path)

print(image_paths)

### load model
model = YOLO('../../datasets/CCTV-Inspection-room/weight/cctv-feed-ml_v3-post-prd-022024_yolov8l.pt')

### run batch
model.predict(image_paths, conf=0.001, project="output_pred",name="post_prd_032024_xxxx",save=True)