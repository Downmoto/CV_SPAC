# Smart Parking Access Control (SPAC)

license plate detection, OCR extraction, resident matching, and access decisions, all from a single CLI.

## final report
The final report can be found in [docs/report/report.pdf](https://github.com/Downmoto/CV_SPAC/blob/main/docs/report/report.pdf)

## setup

```bash
python -m pip install -r requirements.txt
```

for kaggle downloads:

```bash
python -m pip install kaggle
```

## quick start

```bash
python -m src.main --help
```

the cli reads defaults from `configs/default.yaml`. pass action flags to run one or more stages:

```bash
python -m src.main --infer                          # run inference on default test split
python -m src.main --infer --eval --report          # infer + evaluate + report
python -m src.main --all                            # full pipeline end to end
```

## actions

| flag | stage | module |
|---|---|---|
| `--download` | download + unzip kaggle dataset | kaggle cli |
| `--seed-db` | generate sample resident db | `src.data.create_sample_db` |
| `--prepare` | convert raw data to yolo format | `src.data.prepare_kaggle_car_plate_dataset` |
| `--train` | train plate detector | `src.train.train_detector` |
| `--infer` | run end-to-end inference | `src.infer.run_inference` |
| `--eval` | evaluate predictions | `src.eval.evaluate_pipeline` |
| `--report` | generate markdown tables | `src.eval.generate_report_tables` |
| `--all` | run all stages in order | - |

## input overrides

```bash
python -m src.main --infer --image path/to/image.jpg            # single image
python -m src.main --infer --image-dir path/to/images           # directory
python -m src.main --eval --use-ground-truth                    # with ground truth
python -m src.main --eval --ground-truth-csv path/to/gt.csv     # custom gt path
python -m src.main --config other.yaml --infer                  # custom config
```

## outputs

| action | output |
|---|---|
| `--prepare` | `data/processed/car_plate_kaggle/` (yolo splits + `dataset.yaml`) |
| `--train` | `runs/detect/outputs/detector_train/` (weights, curves, `results.csv`) |
| `--infer` | `outputs/predictions/inference_results.json`, `outputs/demo/` |
| `--eval` | `outputs/metrics/evaluation_summary.json`, `evaluation_rows.csv` |
| `--report` | `docs/evaluation_tables.md` |

## project structure

```
src/
  main.py            # cli entrypoint
  data/              # dataset preparation & resident db
  train/             # detector training
  infer/             # inference pipeline & preprocessing export
  eval/              # evaluation & reporting
  ocr/               # multi-backend ocr engine
  matching/          # resident plate matcher
  utils/             # shared helpers
configs/
  default.yaml       # central config for all stages
```

each module under `src/` has its own README with detailed usage.

## config

all default paths and thresholds live in `configs/default.yaml`. prefer editing config over passing long CLI arguments. key sections: `paths`, `dataset`, `training`, `inference`, `ocr`, `evaluation`.

## git and raw images

`data/raw/*` and `data/processed/*` are gitignored, please download datasets locally.

## Notes

- if `models/plate_detector.pt` is missing, inference will fail with a clear error.
- if OCR confidence is low, matching may return denied as expected.
- fuzzy matching can be toggled in `configs/default.yaml`.

## Authors
Arad Fadaei, Sia Tedy
