# src/data

dataset preparation and resident database generation.

## prepare_kaggle_car_plate_dataset

converts the andrewmvd/car-plate-detection kaggle dataset (pascal VOC XML) into YOLO format with train/val/test splits.

```bash
python -m src.data.prepare_kaggle_car_plate_dataset \
  --raw-dir data/raw/car-plate-detection \
  --out-dir data/processed/car_plate_kaggle \
  --clear-out-dir
```

| arg | default |
|---|---|
| `--raw-dir` | `data/raw/car-plate-detection` |
| `--out-dir` | `data/processed/car_plate_kaggle` |
| `--seed` | 42 |
| `--train-ratio` | 0.8 |
| `--val-ratio` | 0.1 |
| `--test-ratio` | 0.1 |
| `--plate-class-name` | `plate` |
| `--clear-out-dir` | flag — wipe output dir first |

**input:** `data/raw/car-plate-detection/{images,annotations}`
**output:** `data/processed/car_plate_kaggle/{images,labels}/{train,val,test}` + `dataset.yaml`

## create_sample_db

generates a synthetic resident database CSV from plate numbers found in ground-truth or inference results.

```bash
python -m src.data.create_sample_db
```

| arg | default |
|---|---|
| `--ground-truth-csv` | `outputs/metrics/ground_truth_template.csv` |
| `--output-csv` | `data/db/residents.csv` |
| `--inference-json` | `outputs/predictions/inference_results.json` |
| `--fallback-to-ocr` | flag — use OCR text when expected_plate is blank |
| `--seed` | 42 |
| `--active-ratio` | 0.85 |

**input:** ground-truth CSV (`expected_plate` column), optionally inference JSON as fallback
**output:** `data/db/residents.csv` with columns: `plate_number`, `resident_name`, `unit`, `vehicle_color`, `status`

## downloading the raw dataset

see [kaggle_download_instructions.md](kaggle_download_instructions.md) for manual download steps, or use:

```bash
python -m src.main --download
```
