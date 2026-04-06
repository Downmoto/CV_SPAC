# dataset preparation for andrewmvd/car-plate-detection

this project now targets:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

the dataset provides:
- images in `images/`
- pascal voc xml annotations in `annotations/`

## quick workflow

1. download dataset to `data/raw/car-plate-detection`
2. convert xml annotations to yolo format and split data:

```bash
/Users/arad/Developer/CV_proj/.venv/bin/python -m src.data.prepare_kaggle_car_plate_dataset \
  --raw-dir data/raw/car-plate-detection \
  --out-dir data/processed/car_plate_kaggle \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --clear-out-dir
```

3. train with generated yaml:

```bash
/Users/arad/Developer/CV_proj/.venv/bin/python -m src.train.train_detector \
  --data-yaml data/processed/car_plate_kaggle/dataset.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640
```

4. copy best weights to `models/plate_detector.pt`

```bash
mkdir -p models
cp runs/detect/outputs/detector_train/weights/best.pt models/plate_detector.pt
```

## generated structure

after conversion, you should have:

```text
data/processed/car_plate_kaggle/
  dataset.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```
