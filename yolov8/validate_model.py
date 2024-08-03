from ultralytics import YOLO
import os

model = YOLO('../asset/cctv-feed-ml_v3-post-prd-022024_yolov8l.pt')

list_step = [0.5, 0.6, 0.7, 0.8]

for x in list_step:
    validation_results = model.val(data='./assets/val.yaml',conf=x)