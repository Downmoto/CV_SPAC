"""extract plate crops from images and create a labeled dataset for crnn training.

uses the trained detector to find plates, then:
- for test images: uses ground-truth expected_plate as label
- for train/val images: uses current ocr engine to generate pseudo-labels
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import yaml

from src.infer.pipeline import PlateDetector, crop_bbox
from src.ocr.ocr_engine import OCREngine
from src.utils.text import normalize_plate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="extract plate crops and build labeled dataset for crnn training"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="data/plate_crops")
    parser.add_argument(
        "--ground-truth-csv",
        default="outputs/metrics/ground_truth_template.csv",
    )
    parser.add_argument("--min-ocr-conf", type=float, default=0.5,
                        help="minimum ocr confidence for pseudo-labels")
    parser.add_argument("--min-plate-len", type=int, default=3,
                        help="minimum plate text length for pseudo-labels")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    ocr_cfg = cfg["ocr"]

    out_dir = Path(args.output_dir)
    crops_dir = out_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth labels for test set
    gt_labels: dict[str, str] = {}
    gt_path = Path(args.ground_truth_csv)
    if gt_path.exists():
        import pandas as pd

        gt_df = pd.read_csv(gt_path)
        for _, row in gt_df.iterrows():
            img_name = Path(str(row["image_path"])).stem
            plate = normalize_plate_text(str(row.get("expected_plate", "")))
            if plate and plate != "UNKNOWN":
                gt_labels[img_name] = plate

    # init detector and ocr
    detector = PlateDetector(
        weights_path=paths["yolo_weights"],
        conf_threshold=float(cfg["inference"]["detector_conf_threshold"]),
    )
    ocr = OCREngine(
        languages=list(ocr_cfg.get("language_list", ["en"])),
        min_conf=0.1,
        backends=list(ocr_cfg.get("backends", ["easyocr", "paddleocr"])),
        use_rectification=bool(ocr_cfg.get("use_rectification", True)),
    )

    # collect all images from train/val/test
    dataset_dir = Path(cfg["dataset"]["out_dir"])
    all_images: list[tuple[str, str]] = []  # (path, split)
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / "images" / split
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                all_images.append((str(img_path), split))

    print(f"total images to process: {len(all_images)}")
    print(f"ground-truth labels available: {len(gt_labels)}")

    rows: list[dict] = []
    idx = 0

    for img_path, split in all_images:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        det = detector.detect(img_bgr)
        crop = crop_bbox(img_bgr, det.bbox_xyxy)
        if crop is None or crop.size == 0:
            continue

        stem = Path(img_path).stem

        # determine label
        if stem in gt_labels:
            label = gt_labels[stem]
            source = "ground_truth"
        else:
            plate_text, conf = ocr.read_plate(crop)
            label = normalize_plate_text(plate_text)
            if not label or conf < args.min_ocr_conf or len(label) < args.min_plate_len:
                continue
            source = "pseudo"

        # save crop
        crop_name = f"{idx:05d}_{stem}.png"
        cv2.imwrite(str(crops_dir / crop_name), crop)

        rows.append({
            "filename": crop_name,
            "label": label,
            "source": source,
            "split": split,
            "original": Path(img_path).name,
        })
        idx += 1

    # write labels csv
    labels_csv = out_dir / "labels.csv"
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "split", "original"])
        writer.writeheader()
        writer.writerows(rows)

    gt_count = sum(1 for r in rows if r["source"] == "ground_truth")
    pseudo_count = sum(1 for r in rows if r["source"] == "pseudo")
    print(f"saved {len(rows)} crops to: {crops_dir}")
    print(f"  ground_truth labels: {gt_count}")
    print(f"  pseudo labels: {pseudo_count}")
    print(f"labels csv: {labels_csv}")


if __name__ == "__main__":
    main()
