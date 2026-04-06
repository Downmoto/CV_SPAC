# Smart Parking Access Control (SPAC)

This repository contains a starter implementation for the SPAC project:
- license plate detection (YOLO-style)
- plate OCR extraction (EasyOCR)
- resident database matching (CSV)
- access decision output (granted/denied)

## unified cli

the project now has a single entrypoint that orchestrates the full workflow.

important:
- use `python -m src.main` (not `python -m src.main.py`)
- the cli reads defaults from `configs/default.yaml`
- you can run one stage at a time or chain stages

quick help:

```bash
python -m src.main --help
```

### why this exists

before this runner, you had to run multiple module-level commands and keep passing many paths manually.
the unified cli solves that by:
- centralizing default paths in config
- using simple action flags (`--prepare`, `--train`, `--infer`, `--eval`, `--report`)
- allowing optional overrides when needed (`--image`, `--image-dir`, `--ground-truth-csv`)

### core actions

each action maps to an existing module:
- `--download` -> kaggle cli download + unzip into `data/raw`
- `--seed-db` -> `src.data.create_sample_db`
- `--prepare` -> `src.data.prepare_kaggle_car_plate_dataset`
- `--train` -> `src.train.train_detector`
- `--extract-crops` -> `src.data.extract_plate_crops`
- `--train-ocr` -> `src.train.train_plate_ocr`
- `--infer` -> `src.infer.run_inference`
- `--eval` -> `src.eval.evaluate_pipeline`
- `--report` -> `src.eval.generate_report_tables`
- `--all` -> runs `prepare -> train -> infer -> eval -> report`

basic action examples:

```bash
# download and unzip raw kaggle dataset
python -m src.main --download

# force fresh re-download
python -m src.main --download --force-download

# create sample resident db
python -m src.main --seed-db

# prepare dataset (kaggle xml -> yolo splits)
python -m src.main --prepare

# train detector
python -m src.main --train

# run inference on default test split from config
python -m src.main --infer

# extract plate crops for crnn training
python -m src.main --extract-crops

# train crnn plate ocr model
python -m src.main --train-ocr

# train crnn with custom epochs and repeat factor
python -m src.main --train-ocr --ocr-epochs 80 --ocr-repeat-factor 5

# run base evaluation
python -m src.main --eval

# run evaluation with ground truth when csv exists
python -m src.main --eval --use-ground-truth

# generate markdown report tables
python -m src.main --report
```

### common combined workflows

train + infer:

```bash
python -m src.main --train --infer
```

infer + evaluate + report:

```bash
python -m src.main --infer --eval --report
```

evaluate with ground truth + report:

```bash
python -m src.main --eval --use-ground-truth --report
```

generate ground-truth template from latest inference output:

```bash
python -m src.eval.create_ground_truth_template \
  --inference-json outputs/predictions/inference_results.json \
  --output-csv outputs/metrics/ground_truth_template.csv
```

full pipeline (long-running, includes training):

```bash
python -m src.main --all
```

download + prepare only:

```bash
python -m src.main --download --prepare
```

### input overrides

the cli uses config defaults, but you can override at runtime.

single image inference:

```bash
python -m src.main --infer --image path/to/image.jpg
```

directory inference:

```bash
python -m src.main --infer --image-dir path/to/images
```

use a custom ground truth file:

```bash
python -m src.main --eval --use-ground-truth --ground-truth-csv path/to/gt.csv
```

use a custom config file:

```bash
python -m src.main --config configs/default.yaml --infer
```

### default outputs produced by actions

- `--prepare`:
  - `data/processed/car_plate_kaggle/dataset.yaml`
  - split images/labels under `data/processed/car_plate_kaggle`
- `--train`:
  - `runs/detect/outputs/detector_train/weights/best.pt`
  - `runs/detect/outputs/detector_train/results.csv`
- `--extract-crops`:
  - `data/plate_crops/images/` (cropped plate images)
  - `data/plate_crops/labels.csv` (filename, label, split)
- `--train-ocr`:
  - `models/plate_crnn.pt` (best crnn weights)
- `--infer`:
  - `outputs/predictions/inference_results.json`
  - visual outputs in `outputs/demo`
- `--eval`:
  - `outputs/metrics/evaluation_summary.json`
  - `outputs/metrics/evaluation_rows.csv` (if ground truth is used)
- `--report`:
  - `docs/evaluation_tables.md`

### config-driven defaults

the following sections in `configs/default.yaml` are used by `src.main`:
- `dataset`: dataset conversion defaults
- `training`: detector training defaults
- `paths`: all key IO paths (weights, jsons, reports)
- `inference`: detection/ocr/matching thresholds
- `evaluation`: positive class label for metrics

if you need to change standard behavior, prefer editing config instead of passing long command arguments.

### recommended day-to-day usage

1. after changing data/splits:
   - run `--prepare`
2. after changing detector settings:
   - run `--train`
3. after changing ocr/matching logic:
   - run `--infer --eval --report`
4. before final submission:
   - run `--eval --use-ground-truth --report`

### troubleshooting

if you get `no action provided`, pass at least one action flag like `--infer`.

if evaluation says ground truth not found:
- check that `outputs/metrics/ground_truth_template.csv` exists, or
- pass `--ground-truth-csv` with a valid file, or
- run base eval without `--use-ground-truth`.

if `--seed-db` says ground truth not found:
- it now auto-generates the template from `outputs/predictions/inference_results.json` when possible,
- if inference json is missing too, run `--infer` first, then re-run `--seed-db`.

if `--seed-db` says expected_plate values are missing:
- either fill `expected_plate` in your ground-truth csv, or
- re-run `--seed-db` with inference available (main flow already enables fallback),
- fallback mode will use `ocr_text` from inference json to build residents when expected plates are blank.

if inference fails because weights are missing:
- train first with `--train`, then copy best weights to `models/plate_detector.pt` if needed.

if you accidentally run `python -m src.main.py ...`:
- switch to `python -m src.main ...`.

if download fails with kaggle auth error:
- configure `~/.kaggle/kaggle.json` first
- set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### git and raw images

raw and processed image folders are intentionally ignored in git:
- `data/raw/*`
- `data/processed/*`

so you can download/unzip datasets locally without committing large image files.

## 1) Environment setup

```bash
python -m pip install -r requirements.txt
```

for kaggle download support:

```bash
python -m pip install kaggle
```

## 2) Create a sample resident database

```bash
python -m src.data.create_sample_db
```

This creates `data/db/residents.csv`.

## 3) Train the detector

prepare the kaggle dataset first (see `src/data/prepare_dataset.md` and `src/data/kaggle_download_instructions.md`).

```bash
python -m src.data.prepare_kaggle_car_plate_dataset \
  --raw-dir data/raw/car-plate-detection \
  --out-dir data/processed/car_plate_kaggle \
  --clear-out-dir
```

then train using the generated dataset yaml:

```bash
python -m src.train.train_detector \
  --data-yaml data/processed/car_plate_kaggle/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640
```

After training, copy best weights to:
- `models/plate_detector.pt`

```bash
mkdir -p models
cp runs/detect/outputs/detector_train/weights/best.pt models/plate_detector.pt
```

## 4) Run end-to-end inference

Single image:

```bash
python -m src.infer.run_inference --image path/to/image.jpg
```

Directory of images:

```bash
python -m src.infer.run_inference --image-dir path/to/images
```

Outputs:
- json: `outputs/predictions/inference_results.json`
- visual results: `outputs/demo`

Export preprocessing steps for one image (for report/presentation figures):

```bash
python -m src.infer.export_preprocessing_steps \
  --image data/processed/car_plate_kaggle/images/test/Cars111.png \
  --output-dir outputs/preprocessing_steps
```

This saves stage images under:
- `outputs/preprocessing_steps/Cars111/01_original.png`
- `outputs/preprocessing_steps/Cars111/02_detected_bbox.png`
- `outputs/preprocessing_steps/Cars111/03_plate_crop.png`
- `outputs/preprocessing_steps/Cars111/04_upscaled.png`
- `outputs/preprocessing_steps/Cars111/05_rectified_or_upscaled.png`
- `outputs/preprocessing_steps/Cars111/06_denoise.png`
- `outputs/preprocessing_steps/Cars111/07_clahe.png`
- `outputs/preprocessing_steps/Cars111/08_otsu_threshold.png`
- `outputs/preprocessing_steps/Cars111/09_adaptive_threshold.png`
- `outputs/preprocessing_steps/Cars111/10_sharpen.png`
- `outputs/preprocessing_steps/Cars111/11_rectified_gray.png`
- `outputs/preprocessing_steps/Cars111/summary.txt`

## 5) Evaluate pipeline outputs

Base metrics from inference output:

```bash
python -m src.eval.evaluate_pipeline \
  --inference-json outputs/predictions/inference_results.json \
  --output-json outputs/metrics/evaluation_summary.json
```

Create ground-truth template for decision-level evaluation:

```bash
python -m src.eval.create_ground_truth_template \
  --inference-json outputs/predictions/inference_results.json \
  --output-csv outputs/metrics/ground_truth_template.csv
```

After filling expected decision and expected plate values, run labeled evaluation:

```bash
python -m src.eval.evaluate_pipeline \
  --inference-json outputs/predictions/inference_results.json \
  --ground-truth-csv outputs/metrics/ground_truth_template.csv \
  --output-json outputs/metrics/evaluation_summary.json \
  --output-csv outputs/metrics/evaluation_rows.csv
```

Generate report-ready markdown tables:

```bash
python -m src.eval.generate_report_tables \
  --detector-results-csv runs/detect/outputs/detector_train/results.csv \
  --evaluation-summary-json outputs/metrics/evaluation_summary.json \
  --output-md docs/evaluation_tables.md
```

## 6) Train CRNN plate OCR model (optional)

the crnn is a lightweight cnn+bilstm+ctc model trained on plate crops extracted from the dataset.
it runs as a third ocr backend alongside easyocr and paddleocr.

extract labeled plate crops (uses ground-truth labels for test, ocr pseudo-labels for train/val):

```bash
python -m src.data.extract_plate_crops \
  --config configs/default.yaml \
  --output-dir data/plate_crops
```

train the crnn model:

```bash
python -m src.train.train_plate_ocr \
  --crops-dir data/plate_crops/images \
  --labels-csv data/plate_crops/labels.csv \
  --output-weights models/plate_crnn.pt \
  --epochs 150 \
  --batch-size 16 \
  --repeat-factor 10
```

for faster cpu training, reduce epochs and repeat factor:

```bash
python -m src.train.train_plate_ocr --epochs 80 --repeat-factor 5
```

once trained, the crnn backend is automatically used during inference if `crnn` is listed in `configs/default.yaml` under `ocr.backends` and `models/plate_crnn.pt` exists.

## 7) Suggested implementation order

1. verify resident db flow with sample csv
2. download and convert andrewmvd kaggle dataset
3. train plate detector and export `models/plate_detector.pt`
4. run inference on validation images
5. evaluate failures and tune thresholds in `configs/default.yaml`

## Notes

- if `models/plate_detector.pt` is missing, inference will fail with a clear error.
- if OCR confidence is low, matching may return denied as expected.
- fuzzy matching can be toggled in `configs/default.yaml`.
