from typing import Any

import cv2
import numpy as np

from src.utils.text import normalize_plate_text

# Default sequential pipeline order — reorder, remove, or insert steps as needed.
# Each name maps to a self-contained preprocessing operation in _seq_step().
SEQUENTIAL_STEPS: list[str] = [
    "upscale",
    "rectify",
    "flatten",
    "grayscale",
    "denoise",
    "clahe",
    "sharpen",
    "binarize",
]


class OCREngine:
    def __init__(
        self,
        languages: list[str] | None = None,
        min_conf: float = 0.35,
        backends: list[str] | None = None,
        use_rectification: bool = True,
        gpu: bool = True,
        sequential_steps: list[str] | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.min_conf = min_conf
        self.backends = [b.lower() for b in (backends or ["easyocr", "paddleocr"])]
        self.use_rectification = use_rectification
        self.gpu = gpu
        self.sequential_steps: list[str] = list(sequential_steps or SEQUENTIAL_STEPS)
        self._easy_reader = None
        self._paddle_reader = None

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

    def _flatten_plate(self, image_bgr: np.ndarray) -> np.ndarray:
        """correct perspective distortion by finding the plate quadrilateral
        and warping it to a flat rectangle."""
        h, w = image_bgr.shape[:2]
        if h < 10 or w < 10:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120)

        # aggressively close gaps so text merges into the plate body
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 0.1 * h * w:
            return image_bgr

        # try progressively looser epsilon to find a 4-corner approximation
        peri = cv2.arcLength(largest, True)
        quad = None
        for eps_factor in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
            approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
            if len(approx) == 4:
                quad = approx
                break

        # fallback: use the 4 corners of the minimum area bounding rect
        if quad is None:
            rect = cv2.minAreaRect(largest)
            quad = cv2.boxPoints(rect).astype(np.int32)

        pts = quad.reshape(4, 2).astype(np.float32)

        # order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.array([
            pts[np.argmin(s)],   # top-left
            pts[np.argmin(d)],   # top-right
            pts[np.argmax(s)],   # bottom-right
            pts[np.argmax(d)],   # bottom-left
        ], dtype=np.float32)

        # compute output dimensions
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_left = np.linalg.norm(ordered[3] - ordered[0])
        h_right = np.linalg.norm(ordered[2] - ordered[1])
        out_w = int(max(w_top, w_bot))
        out_h = int(max(h_left, h_right))

        if out_w < 10 or out_h < 10:
            return image_bgr

        # skip if the warp would barely change anything (nearly rectangular already)
        src_rect_area = out_w * out_h
        if src_rect_area > 0 and abs(area - src_rect_area) / src_rect_area < 0.03:
            return image_bgr

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        mat = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image_bgr, mat, (out_w, out_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped

    def _seq_step(self, name: str, img: np.ndarray) -> np.ndarray:
        """Dispatch a single named preprocessing step.

        Each step is self-contained: it inspects whether its input is
        grayscale (2-D) or BGR (3-D) and converts as needed, so steps
        can be freely reordered in ``self.sequential_steps``.
        """
        is_gray = img.ndim == 2

        if name == "upscale":
            return self._upscale(img)

        if name == "rectify":
            if not self.use_rectification:
                return img
            bgr = img if not is_gray else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return self._rectify_plate(bgr)

        if name == "flatten":
            bgr = img if not is_gray else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return self._flatten_plate(bgr)

        if name == "grayscale":
            return img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if name == "denoise":
            return cv2.bilateralFilter(img, 9, 75, 75)

        if name == "clahe":
            g = img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)

        if name == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            return cv2.filter2D(img, -1, kernel)

        if name == "binarize":
            g = img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        if name == "adaptive_threshold":
            g = img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10,
            )

        raise ValueError(f"Unknown preprocessing step: {name!r}")

    def _run_sequential_pipeline(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run every step in ``self.sequential_steps`` in order,
        feeding each step's output into the next."""
        img = image_bgr
        for step in self.sequential_steps:
            img = self._seq_step(step, img)
        # ensure 3-channel output for OCR readers
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _prepare_variants(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        up = self._upscale(image_bgr)
        rectified = self._rectify_plate(up) if self.use_rectification else up
        flattened = self._flatten_plate(rectified)
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

        sequential = self._run_sequential_pipeline(image_bgr)

        variants = [
            up,
            rectified,
            flattened,
            cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR),
            sequential,
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
