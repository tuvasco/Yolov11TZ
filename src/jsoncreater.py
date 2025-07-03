import glob, os, json
import random

IM = 'data/yolodataset/images/train_unlabeled/'
LB = 'data/yolodataset/images/train_unlabeled/' 

tasks = []
for img_path in glob.glob(IM + '*.jpg'):
    label_path = img_path.replace('.jpg', '.txt')
    imgname = os.path.basename(img_path)
    results = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for i, line in enumerate(f):
                cls, xc, yc, w, h = map(float, line.split())
                results.append({
                    "id": f"{imgname}-{i}",
                    "type": "rectanglelabels",
                    "from_name": "tag",
                    "to_name": "img",
                    "original_width": None,
                    "original_height": None,
                    "value": {
                        "x": xc*100, "y": yc*100,
                        "width": w*100, "height": h*100,
                        "rotation": 0,
                        "rectanglelabels": [f"class_{int(cls)}"]
                    }
                })
    task = {
        "data": {"img": img_path},
        "predictions": [{
            "model_version": "yolo-custom",
            "result": results,
            "score": 1.0
        }]
    }
    tasks.append(task)
with open('prelabel_tasks.json', 'w') as f:
    json.dump(tasks, f)
print("создан JSON с", len(tasks), "тасками")
