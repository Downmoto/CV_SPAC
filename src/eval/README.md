# src/eval

evaluation, ground-truth management, and report generation.

## evaluate_pipeline

computes base metrics (detection rate, OCR fill rate, mean confidences) and optionally decision-level accuracy/precision/recall/F1 when ground-truth is provided.

```bash
# base metrics only
python -m src.eval.evaluate_pipeline \
  --inference-json outputs/predictions/inference_results.json

# with ground truth
python -m src.eval.evaluate_pipeline \
  --inference-json outputs/predictions/inference_results.json \
  --ground-truth-csv outputs/metrics/ground_truth_template.csv \
  --output-csv outputs/metrics/evaluation_rows.csv
```

| arg | default |
|---|---|
| `--inference-json` | `outputs/predictions/inference_results.json` |
| `--ground-truth-csv` | None |
| `--output-json` | `outputs/metrics/evaluation_summary.json` |
| `--output-csv` | `outputs/metrics/evaluation_rows.csv` |
| `--decision-positive-label` | `Access Granted` |

## create_ground_truth_template

generates a blank CSV template from inference results for manual labeling.

```bash
python -m src.eval.create_ground_truth_template \
  --inference-json outputs/predictions/inference_results.json \
  --output-csv outputs/metrics/ground_truth_template.csv
```

| arg | default |
|---|---|
| `--inference-json` | `outputs/predictions/inference_results.json` |
| `--output-csv` | `outputs/metrics/ground_truth_template.csv` |
| `--default-decision` | `Access Denied` |

**output:** CSV with columns: `image_path`, `expected_decision`, `expected_plate`, `notes`

## generate_report_tables

combines detector training metrics and pipeline evaluation into markdown tables.

```bash
python -m src.eval.generate_report_tables \
  --detector-results-csv runs/detect/outputs/detector_train/results.csv \
  --evaluation-summary-json outputs/metrics/evaluation_summary.json \
  --output-md docs/evaluation_tables.md
```

| arg | default |
|---|---|
| `--detector-results-csv` | `runs/detect/outputs/detector_train/results.csv` |
| `--evaluation-summary-json` | `outputs/metrics/evaluation_summary.json` |
| `--output-md` | `docs/evaluation_tables.md` |
