import argparse
from pathlib import Path

import cv2

from src.infer.pipeline import PlateDetector, crop_bbox, draw_result
from src.ocr.ocr_engine import OCREngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="export image processing steps for one image"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="input image path",
    )
    parser.add_argument(
        "--weights",
        default="models/plate_detector.pt",
        help="detector weights path",
    )
    parser.add_argument(
        "--detector-conf-threshold",
        type=float,
        default=0.25,
        help="detector confidence threshold",
    )
    parser.add_argument(
        "--ocr-conf-threshold",
        type=float,
        default=0.35,
        help="ocr confidence threshold",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/preprocessing_steps",
        help="output folder root",
    )
    parser.add_argument(
        "--no-rectification",
        action="store_true",
        help="disable geometric rectification variant",
    )
    return parser.parse_args()


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    detector = PlateDetector(
        weights_path=args.weights,
        conf_threshold=args.detector_conf_threshold,
    )
    det = detector.detect(image_bgr)

    out_dir = Path(args.output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(out_dir / "01_original.png", image_bgr)

    result = {
        "detector": {
            "bbox_xyxy": det.bbox_xyxy,
            "conf": det.conf,
        },
        "ocr": {
            "plate_text": "",
            "confidence": 0.0,
        },
        "decision": {
            "label": "",
            "matched": False,
            "matched_plate": None,
            "match_score": 0.0,
        },
    }
    boxed = draw_result(image_bgr, result)
    save_image(out_dir / "02_detected_bbox.png", boxed)

    crop = crop_bbox(image_bgr, det.bbox_xyxy)
    if crop is None:
        raise RuntimeError("detector did not return a valid bbox for this image")
    save_image(out_dir / "03_plate_crop.png", crop)

    ocr = OCREngine(
        languages=["en"],
        min_conf=args.ocr_conf_threshold,
        backends=["easyocr", "paddleocr"],
        use_rectification=not args.no_rectification,
    )

    variants = ocr._prepare_variants(crop)
    variant_names = [
        "04_upscaled",
        "05_rectified_or_upscaled",
        "06_flattened",
        "07_denoise",
        "08_clahe",
        "09_otsu_threshold",
        "10_adaptive_threshold",
        "11_sharpen",
        "12_rectified_gray",
        "13_sequential_pipeline",
    ]

    for name, variant in zip(variant_names, variants):
        save_image(out_dir / f"{name}.png", variant)

    final_text, final_conf = ocr.read_plate(crop)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"image: {image_path}",
                f"bbox_xyxy: {det.bbox_xyxy}",
                f"detector_conf: {det.conf:.6f}",
                f"final_ocr_text: {final_text}",
                f"final_ocr_conf: {final_conf:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved preprocessing steps to: {out_dir}")


if __name__ == "__main__":
    main()
