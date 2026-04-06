import argparse
import json
from pathlib import Path

import cv2
import yaml

from src.infer.pipeline import SPACPipeline, draw_result


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run spac pipeline on images")
    parser.add_argument("--config", default="configs/default.yaml", help="path to yaml config")
    parser.add_argument("--image", help="single image path")
    parser.add_argument("--image-dir", help="directory with images")
    return parser.parse_args()


def collect_images(single_image: str | None, image_dir: str | None) -> list[str]:
    if single_image:
        return [single_image]
    if image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [
            str(p)
            for p in sorted(Path(image_dir).iterdir())
            if p.suffix.lower() in exts and p.is_file()
        ]
    raise ValueError("provide --image or --image-dir")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    infer_cfg = cfg["inference"]
    ocr_cfg = cfg["ocr"]

    pipeline = SPACPipeline(
        weights_path=paths["yolo_weights"],
        resident_db_csv=paths["resident_db_csv"],
        detector_conf_threshold=float(infer_cfg["detector_conf_threshold"]),
        ocr_conf_threshold=float(infer_cfg["ocr_conf_threshold"]),
        use_fuzzy_matching=bool(infer_cfg["use_fuzzy_matching"]),
        fuzzy_match_threshold=int(infer_cfg["fuzzy_match_threshold"]),
        ocr_languages=list(ocr_cfg.get("language_list", ["en"])),
        ocr_backends=list(ocr_cfg.get("backends", ["easyocr", "paddleocr"])),
        ocr_use_rectification=bool(ocr_cfg.get("use_rectification", True)),
    )

    image_paths = collect_images(args.image, args.image_dir)
    output_json = Path(paths["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_visual_dir = Path(paths["output_visual_dir"])
    output_visual_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for image_path in image_paths:
        result = pipeline.run_on_image(image_path)
        all_results.append(result)

        image_bgr = cv2.imread(image_path)
        vis = draw_result(image_bgr, result)
        vis_path = output_visual_dir / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(vis_path), vis)

        print(
            f"{Path(image_path).name}: {result['decision']['label']} "
            f"(plate={result['ocr']['plate_text']}, "
            f"score={result['decision']['match_score']:.1f})"
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"saved json results to: {output_json}")
    print(f"saved visual outputs to: {output_visual_dir}")


if __name__ == "__main__":
    main()
