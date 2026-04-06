# src/train

model training modules.

## train_detector

thin wrapper around ultralytics YOLO training for the license plate detector.

```bash
python -m src.train.train_detector \
  --data-yaml data/processed/car_plate_kaggle/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50
```

| arg | default |
|---|---|
| `--data-yaml` | **(required)** |
| `--model` | `yolov8n.pt` |
| `--epochs` | 50 |
| `--imgsz` | 640 |
| `--batch` | 16 |
| `--device` | auto |
| `--project` | `outputs` |
| `--name` | `detector_train` |

**input:** `dataset.yaml` + base YOLO weights
**output:** `runs/detect/outputs/detector_train/` (weights, curves, `results.csv`)

after training, copy best weights:

```bash
cp runs/detect/outputs/detector_train/weights/best.pt models/plate_detector.pt
```
