from ultralytics import YOLO

model = YOLO('models/yolo-custom.pt')
model.predict(
    source='data/yolodataset/images/train_unlabeled',
    save_txt=True, 
    conf=0.5    
)
