import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SplitCounts:
    train: int
    val: int
    test: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="prepare andrewmvd car-plate-detection kaggle dataset for yolo"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/car-plate-detection",
        help="dataset root with images/ and annotations/",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/car_plate_kaggle",
        help="output root for yolo dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--plate-class-name",
        default="plate",
        help="class name to use in generated dataset yaml",
    )
    parser.add_argument(
        "--clear-out-dir",
        action="store_true",
        help="delete output dir before writing",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_yolo_line(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> str:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    bbox_w = (xmax - xmin) / width
    bbox_h = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"


def parse_voc_xml(xml_path: Path) -> tuple[str, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    if filename_node is None or filename_node.text is None:
        raise ValueError(f"missing filename in {xml_path}")
    filename = filename_node.text.strip()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"missing size node in {xml_path}")

    width_node = size_node.find("width")
    height_node = size_node.find("height")
    if width_node is None or height_node is None:
        raise ValueError(f"missing width/height in {xml_path}")

    width = int(float(width_node.text or 0))
    height = int(float(height_node.text or 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {xml_path}: {width}x{height}")

    lines: list[str] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin_node = bnd.find("xmin")
        ymin_node = bnd.find("ymin")
        xmax_node = bnd.find("xmax")
        ymax_node = bnd.find("ymax")
        if (
            xmin_node is None
            or ymin_node is None
            or xmax_node is None
            or ymax_node is None
        ):
            continue

        xmin = float(xmin_node.text or 0)
        ymin = float(ymin_node.text or 0)
        xmax = float(xmax_node.text or 0)
        ymax = float(ymax_node.text or 0)

        xmin = max(0.0, min(xmin, float(width)))
        xmax = max(0.0, min(xmax, float(width)))
        ymin = max(0.0, min(ymin, float(height)))
        ymax = max(0.0, min(ymax, float(height)))

        if xmax <= xmin or ymax <= ymin:
            continue

        lines.append(to_yolo_line(xmin, ymin, xmax, ymax, width, height))

    return filename, lines


def split_filenames(items: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[str]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train : n_train + n_val]
    test_items = shuffled[n_train + n_val :]
    return {"train": train_items, "val": val_items, "test": test_items}


def write_dataset_yaml(out_dir: Path, class_name: str) -> Path:
    yaml_path = out_dir / "dataset.yaml"
    payload = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return yaml_path


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    raw_dir = Path(args.raw_dir)
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"

    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(
            "expected dataset layout: <raw-dir>/images and <raw-dir>/annotations"
        )

    out_dir = Path(args.out_dir)
    if args.clear_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / split)
        ensure_dir(out_dir / "labels" / split)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"no xml annotations found in {annotations_dir}")

    parsed: dict[str, tuple[Path, list[str]]] = {}
    for xml_path in xml_files:
        filename, lines = parse_voc_xml(xml_path)
        image_path = images_dir / filename
        if not image_path.exists():
            fallback_candidates = list(images_dir.glob(f"{Path(filename).stem}.*"))
            if not fallback_candidates:
                raise FileNotFoundError(
                    f"image file '{filename}' referenced by {xml_path.name} not found"
                )
            image_path = fallback_candidates[0]

        parsed[image_path.name] = (image_path, lines)

    split_map = split_filenames(
        list(parsed.keys()), args.train_ratio, args.val_ratio, args.seed
    )

    for split_name, names in split_map.items():
        for img_name in names:
            image_path, yolo_lines = parsed[img_name]
            dst_image = out_dir / "images" / split_name / image_path.name
            dst_label = out_dir / "labels" / split_name / f"{image_path.stem}.txt"

            shutil.copy2(image_path, dst_image)
            with dst_label.open("w", encoding="utf-8") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines) + "\n")

    dataset_yaml = write_dataset_yaml(out_dir, args.plate_class_name)

    counts = SplitCounts(
        train=len(split_map["train"]),
        val=len(split_map["val"]),
        test=len(split_map["test"]),
    )

    print(f"prepared dataset at: {out_dir}")
    print(f"dataset yaml: {dataset_yaml}")
    print(f"splits -> train: {counts.train}, val: {counts.val}, test: {counts.test}")


if __name__ == "__main__":
    main()
