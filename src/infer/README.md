# src/infer

inference pipeline and diagnostic tools.

## run_inference

runs the full detect → OCR → match → decision pipeline on one image or a directory.

```bash
python -m src.infer.run_inference --image path/to/image.jpg
python -m src.infer.run_inference --image-dir path/to/images
```

| arg | default |
|---|---|
| `--config` | `configs/default.yaml` |
| `--image` | None (single image) |
| `--image-dir` | None (batch) |
| `--device` | auto |

**output:** `outputs/predictions/inference_results.json`, annotated images in `outputs/demo/`

## pipeline

library module — not invoked directly. contains:

- `PlateDetector` — lazy-loaded YOLO wrapper
- `SPACPipeline` — orchestrates detector + OCR + matcher
- `crop_bbox()` — padded plate crop from detection bbox
- `draw_result()` — annotated image with bbox + label overlay

## export_preprocessing_steps

diagnostic tool that saves every intermediate preprocessing image for a single input.

```bash
python -m src.infer.export_preprocessing_steps \
  --image data/processed/car_plate_kaggle/images/test/Cars111.png
```

| arg | default |
|---|---|
| `--image` | **(required)** |
| `--weights` | `models/plate_detector.pt` |
| `--detector-conf-threshold` | 0.25 |
| `--ocr-conf-threshold` | 0.35 |
| `--output-dir` | `outputs/preprocessing_steps` |
| `--no-rectification` | flag |

**output:** numbered images (`01_original.png` … `13_sequential_pipeline.png`) + `summary.txt` under `outputs/preprocessing_steps/{stem}/`
