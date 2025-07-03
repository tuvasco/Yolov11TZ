import cv2
import albumentations as A
import os
import glob

IMAGES_DIR = 'data/yolodataset/images/train'
LABELS_DIR = 'data/yolodataset/labels/train'
AUG_IMAGES_DIR = 'data/yolodataset/images/train_aug'
AUG_LABELS_DIR = 'data/yolodataset/labels/train_aug'

all_images = glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_labels(label_path):
    bboxes = []
    class_labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, *box = map(float, parts)
                    class_labels.append(int(cls))
                    bboxes.append(box)
    except FileNotFoundError:
        print(f'Файл меток не найден: {label_path}')
    return bboxes, class_labels

def save_labels(label_path, class_labels, bboxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for label, box in zip(class_labels, bboxes):
            box_str = ' '.join(map(str, box))
            f.write(f"{label} {box_str}\n")

def augment_and_save(image_path):
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    img = cv2.imread(image_path)
    if img is None:
        print(f'Не удалось загрузить изображение: {image_path}')
        return
    bboxes, class_labels = read_labels(label_path)
    if not bboxes:
        print(f'Нет меток для изображения: {image_path}')
        return
    
    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
  
    rel_path = os.path.relpath(image_path, IMAGES_DIR)
    base, ext = os.path.splitext(rel_path)
    new_img_path = os.path.join(AUG_IMAGES_DIR, f"{base}_augment{ext}")
    new_label_path = os.path.join(AUG_LABELS_DIR, f"{base}_augment.txt")
    os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
    cv2.imwrite(new_img_path, augmented['image'])
    save_labels(new_label_path, augmented['class_labels'], augmented['bboxes'])

if __name__ == '__main__':
    for image_path in all_images:
        augment_and_save(image_path)
