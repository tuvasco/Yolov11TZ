from ultralytics import YOLO

model = YOLO("models/yolo-best.pt")

model.predict(
    source="data/raw_videos/2_1.MOV",
    save=True,
    imgsz=640,
    device=0,
    iou=0.5, 
)