from ultralytics import YOLO

model = YOLO("yolov8n.pt")
train_results = model.train(data="coco81.yaml",epochs=100, device=0) 