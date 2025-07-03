ЗАПУСК YOLO

# 1. Создать окружение из YAML
conda env create -f environment.yaml

# 2. Активировать его
conda activate yolov11_seg

# 3.Положить в папку raw_videos ваше видео
# 4.Открыть src/predict.py, и заменить в пути к файлу название на ваш файл
# 4.В терминале вызвать скрипт [python src/predict.py]
