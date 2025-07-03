import os
from collections import Counter

labels_dir = "D:/YoloTZ/data/yolodataset/labels/train"  

counter = Counter()

for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(labels_dir, filename), "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                counter[class_id] += 1

print("Количество объектов каждого класса:")
for class_id, count in counter.items():
    print(f"Класс {class_id}: {count}")