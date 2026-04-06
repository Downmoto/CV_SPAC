from typing import Any

import cv2
import numpy as np

from src.utils.text import normalize_plate_text


class OCREngine:
    def __init__(
        self,
        languages: list[str] | None = None,
        min_conf: float = 0.35,
        backends: list[str] | None = None,
        use_rectification: bool = True,
        gpu: bool = True,
        crnn_weights: str | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.min_conf = min_conf
        self.backends = [b.lower() for b in (backends or ["easyocr", "paddleocr"])]
        self.use_rectification = use_rectification
        self.gpu = gpu
        self._easy_reader = None
        self._paddle_reader = None
        self._crnn_model = None
        self._crnn_weights = crnn_weights

    def _plate_plausibility(self, text: str) -> float:
        if not text:
            return -1.0

        length = len(text)
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        unique = len(set(text))

        score = 0.0

        # length scoring: strongly reward plate-like lengths, harshly penalize very short
        if 5 <= length <= 10:
            score += 0.30
        elif length in (4, 11):
            score += 0.15
        elif length == 3:
            score -= 0.15
        elif length <= 2:
            score -= 0.50

        # mixed alpha+digit is a strong plate signal
        if letters > 0 and digits > 0:
            score += 0.20
        elif length >= 5:
            score -= 0.05

        # repeated characters with low uniqueness
        if length >= 5 and unique <= 2:
            score -= 0.15

        return score

    def _candidate_rank(self, text: str, conf: float) -> tuple[float, float, int]:
        return (conf + self._plate_plausibility(text), conf, len(text))

    def _ensure_easy_reader(self) -> Any:
        if self._easy_reader is None:
            import easyocr

            self._easy_reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._easy_reader

    def _ensure_paddle_reader(self) -> Any:
        if self._paddle_reader is None:
            from paddleocr import PaddleOCR

            self._paddle_reader = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=self.gpu,
            )
        return self._paddle_reader

    def _ensure_crnn_model(self):
        if self._crnn_model is None:
            import torch
            from src.ocr.plate_crnn import PlateRecCRNN, NUM_CLASSES

            self._crnn_model = PlateRecCRNN(img_h=32, num_classes=NUM_CLASSES, hidden_size=128)
            weights_path = self._crnn_weights or "models/plate_crnn.pt"
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._crnn_model.load_state_dict(state)
            self._crnn_model.eval()
        return self._crnn_model

    def _read_crnn(self, image_bgr: np.ndarray) -> list[tuple[str, float]]:
        """run crnn plate recognition on a single image variant."""
        import torch
        from src.ocr.plate_crnn import decode_output

        model = self._ensure_crnn_model()

        # preprocess: grayscale, resize to 32xW keeping aspect, pad to 128
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        h, w = gray.shape[:2]
        img_h, img_w = 32, 128
        ratio = img_h / max(1, h)
        new_w = min(int(w * ratio), img_w)
        resized = cv2.resize(gray, (new_w, img_h), interpolation=cv2.INTER_CUBIC)
        if new_w < img_w:
            pad = np.zeros((img_h, img_w - new_w), dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=1)

        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(tensor)  # (seq_len, 1, classes)

        # greedy decode with confidence
        probs = output.squeeze(1).exp()  # (seq_len, classes)
        max_probs, preds = probs.max(1)  # (seq_len,)

        text = decode_output(preds.tolist())
        # confidence: average of non-blank character probabilities
        non_blank = [(p.item(), idx.item()) for p, idx in zip(max_probs, preds) if idx.item() != 0]
        if non_blank:
            conf = sum(p for p, _ in non_blank) / len(non_blank)
        else:
            conf = 0.0

        text = normalize_plate_text(text)
        candidates = []
        if text:
            candidates.append((text, conf))
        return candidates

    def _upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        min_side = min(h, w)
        if min_side >= 80:
            scale = 2.0
        else:
            scale = 3.0
        return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _rectify_plate(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return image_bgr

        rect = cv2.minAreaRect(largest)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 1 or rh <= 1:
            return image_bgr

        if angle < -45:
            angle += 90
        rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            image_bgr,
            rot,
            (image_bgr.shape[1], image_bgr.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        x, y, w, h = cv2.boundingRect(largest)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(rotated.shape[1], x + w)
        y1 = min(rotated.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return rotated
        crop = rotated[y0:y1, x0:x1]
        if crop.size == 0:
            return rotated

        target_h = 96
        scale = target_h / max(1, crop.shape[0])
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized

    def _prepare_variants(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        up = self._upscale(image_bgr)
        rectified = self._rectify_plate(up) if self.use_rectification else up
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

        denoise = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoise)

        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen = cv2.filter2D(clahe, -1, kernel)

        variants = [
            up,
            rectified,
            cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR),
        ]
        return variants

    def _read_best(self, image_bgr: np.ndarray) -> list[tuple[str, float]]:
        # returns multiple candidates: single best box + concatenated full plate
        reader = self._ensure_easy_reader()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(
            image_rgb,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.2,
        )

        candidates: list[tuple[str, float]] = []
        if not results:
            return candidates

        # single best box
        best_text = ""
        best_conf = 0.0
        for _, text, conf in results:
            score = float(conf)
            candidate = normalize_plate_text(str(text))
            if score > best_conf:
                best_text = candidate
                best_conf = score
            elif score == best_conf and len(candidate) > len(best_text):
                best_text = candidate
        if best_text:
            candidates.append((best_text, best_conf))

        # concatenated: sort boxes left-to-right by x midpoint, join text
        if len(results) > 1:
            sorted_boxes = sorted(
                results, key=lambda r: (r[0][0][0] + r[0][2][0]) / 2
            )
            joined = "".join(normalize_plate_text(str(r[1])) for r in sorted_boxes)
            avg_conf = sum(float(r[2]) for r in sorted_boxes) / len(sorted_boxes)
            if joined and joined != best_text:
                candidates.append((joined, avg_conf))

        return candidates

    def _read_best_paddle(self, image_bgr: np.ndarray) -> list[tuple[str, float]]:
        # returns multiple candidates: single best box + concatenated full plate
        reader = self._ensure_paddle_reader()
        results = reader.ocr(image_bgr, cls=True)

        candidates: list[tuple[str, float]] = []
        if not results:
            return candidates

        lines = results[0] if isinstance(results, list) and len(results) > 0 else []
        parsed: list[tuple[list, str, float]] = []
        best_text = ""
        best_conf = 0.0

        for item in lines:
            if not item or len(item) < 2:
                continue
            text_conf = item[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = str(text_conf[0])
            conf = float(text_conf[1])
            candidate = normalize_plate_text(text)
            parsed.append((item[0], candidate, conf))
            if conf > best_conf:
                best_text = candidate
                best_conf = conf
            elif conf == best_conf and len(candidate) > len(best_text):
                best_text = candidate

        if best_text:
            candidates.append((best_text, best_conf))

        # concatenated: sort by x midpoint of bounding polygon, join text
        if len(parsed) > 1:
            sorted_boxes = sorted(
                parsed,
                key=lambda p: sum(pt[0] for pt in p[0]) / max(1, len(p[0])),
            )
            joined = "".join(p[1] for p in sorted_boxes)
            avg_conf = sum(p[2] for p in sorted_boxes) / len(sorted_boxes)
            if joined and joined != best_text:
                candidates.append((joined, avg_conf))

        return candidates

    def read_plate(self, image_bgr) -> tuple[str, float]:
        if image_bgr is None:
            return "", 0.0

        variants = self._prepare_variants(image_bgr)
        best_text = ""
        best_conf = 0.0
        best_rank = self._candidate_rank("", 0.0)

        for variant in variants:
            candidates: list[tuple[str, float]] = []

            if "easyocr" in self.backends:
                try:
                    candidates.extend(self._read_best(variant))
                except Exception:
                    pass

            if "paddleocr" in self.backends:
                try:
                    candidates.extend(self._read_best_paddle(variant))
                except Exception:
                    pass

            if "crnn" in self.backends:
                try:
                    candidates.extend(self._read_crnn(variant))
                except Exception:
                    pass

            if not candidates:
                continue

            for text, conf in candidates:
                rank = self._candidate_rank(text, conf)
                if rank > best_rank:
                    best_text = text
                    best_conf = conf
                    best_rank = rank

        if best_conf < self.min_conf:
            return best_text, best_conf

        return best_text, best_conf
