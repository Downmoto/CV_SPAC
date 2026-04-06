from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from src.matching.matcher import MatchResult, ResidentMatcher
from src.ocr.ocr_engine import OCREngine


@dataclass
class DetectionResult:
    bbox_xyxy: tuple[int, int, int, int] | None
    conf: float


class PlateDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.25, device: str | None = None) -> None:
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if not Path(self.weights_path).exists():
                raise FileNotFoundError(
                    f"detector weights not found at: {self.weights_path}. "
                    "train the detector first or update configs/default.yaml"
                )
            from ultralytics import YOLO

            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, image_bgr) -> DetectionResult:
        model = self._ensure_model()
        preds = model.predict(source=image_bgr, conf=self.conf_threshold, device=self.device, verbose=False)
        if not preds:
            return DetectionResult(None, 0.0)

        boxes = preds[0].boxes
        if boxes is None or boxes.conf is None or boxes.xyxy is None or len(boxes) == 0:
            return DetectionResult(None, 0.0)

        best_idx = int(boxes.conf.argmax().item())
        conf = float(boxes.conf[best_idx].item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        return DetectionResult((int(x1), int(y1), int(x2), int(y2)), conf)


def crop_bbox(
    image_bgr,
    bbox_xyxy: tuple[int, int, int, int] | None,
    pad_x_ratio: float = 0.14,
    pad_y_ratio: float = 0.22,
):
    if bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image_bgr.shape[:2]

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_x_ratio))
    pad_y = int(round(bh * pad_y_ratio))

    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


class SPACPipeline:
    def __init__(
        self,
        weights_path: str,
        resident_db_csv: str,
        detector_conf_threshold: float = 0.25,
        ocr_conf_threshold: float = 0.35,
        use_fuzzy_matching: bool = True,
        fuzzy_match_threshold: int = 90,
        ocr_languages: list[str] | None = None,
        ocr_backends: list[str] | None = None,
        ocr_use_rectification: bool = True,
        device: str | None = None,
        crnn_weights: str | None = None,
    ) -> None:
        self.detector = PlateDetector(weights_path, detector_conf_threshold, device=device)
        self.ocr = OCREngine(
            ocr_languages or ["en"],
            ocr_conf_threshold,
            backends=ocr_backends or ["easyocr", "paddleocr"],
            use_rectification=ocr_use_rectification,
            gpu=device is not None and str(device) != "cpu",
            crnn_weights=crnn_weights,
        )
        self.matcher = ResidentMatcher(
            resident_db_csv,
            use_fuzzy_matching=use_fuzzy_matching,
            fuzzy_match_threshold=fuzzy_match_threshold,
        )

    def run_on_image(self, image_path: str) -> dict[str, Any]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")

        det = self.detector.detect(image_bgr)
        crop = crop_bbox(image_bgr, det.bbox_xyxy)
        plate_text, ocr_conf = self.ocr.read_plate(crop)
        match_result: MatchResult = self.matcher.match(plate_text)

        result = {
            "image_path": image_path,
            "detector": asdict(det),
            "ocr": {
                "plate_text": plate_text,
                "confidence": ocr_conf,
            },
            "decision": {
                "label": match_result.decision,
                "matched": match_result.matched,
                "matched_plate": match_result.matched_plate,
                "match_score": match_result.score,
            },
            "record": match_result.record,
        }
        return result


def draw_result(image_bgr, result: dict[str, Any]):
    out = image_bgr.copy()
    bbox = result["detector"]["bbox_xyxy"]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = result["decision"]["label"]
    plate = result["ocr"]["plate_text"] or "UNKNOWN"
    text = f"{label} | plate={plate}"
    cv2.putText(out, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out
