from ultralytics import YOLO


def main():
    model = YOLO('models/yolo-custom.pt')
    model.train(
        data='data/yolodataset/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        workers=6,
        multi_scale=True,
        device=0
    )

if __name__ == "__main__":
    main()