# src/ocr

multi-backend OCR engine for license plate text extraction.

## OCREngine

library module — no CLI. used by the inference pipeline.

```python
from src.ocr.ocr_engine import OCREngine

ocr = OCREngine(
    languages=["en"],
    min_conf=0.35,
    backends=["easyocr", "paddleocr"],
    use_rectification=True,
    sequential_steps=None,  # uses SEQUENTIAL_STEPS default
)

text, confidence = ocr.read_plate(plate_crop_bgr)
```

### how it works

1. generates multiple preprocessing **variants** of the plate crop (upscaled, rectified, flattened, denoised, CLAHE, binarized, sharpened, etc.)
2. includes a **sequential pipeline** variant that chains steps in order (configurable)
3. runs each variant through all enabled OCR backends (easyocr, paddleocr)
4. ranks all candidates by confidence + plate plausibility heuristic
5. returns the best (text, confidence) pair

### sequential pipeline

the sequential pipeline feeds each step's output into the next. the default order is:

```
upscale → rectify → flatten → grayscale → denoise → clahe → sharpen → binarize
```

reorder by passing a custom list:

```python
ocr = OCREngine(sequential_steps=["upscale", "flatten", "grayscale", "clahe", "binarize"])
```

available steps: `upscale`, `rectify`, `flatten`, `grayscale`, `denoise`, `clahe`, `sharpen`, `binarize`, `adaptive_threshold`

### config

controlled via `configs/default.yaml` under the `ocr` section:

- `language_list` — OCR languages
- `backends` — list of backends to use
- `use_rectification` — enable geometric rectification
