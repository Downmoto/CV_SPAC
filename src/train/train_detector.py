import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train yolo detector for license plate")
    parser.add_argument("--data-yaml", required=True, help="dataset yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="training device (e.g., 0 for first GPU, cpu for CPU, or 0,1 for multi-GPU)",
    )
    parser.add_argument("--project", default="runs/detect/outputs")
    parser.add_argument("--name", default="detector_train")
    parser.add_argument(
        "--export-weights",
        default="models/plate_detector.pt",
        help="path to copy trained best checkpoint to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    save_dir = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None) is not None:
        save_dir = Path(trainer.save_dir)
    else:
        results_save_dir = getattr(results, "save_dir", None)
        if results_save_dir is not None:
            save_dir = Path(results_save_dir)

    if save_dir is None:
        raise RuntimeError("could not determine training output directory to locate best.pt")

    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best checkpoint not found: {best_weights}")

    export_path = Path(args.export_weights)
    if export_path.suffix == "":
        export_path = export_path.with_suffix(".pt")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, export_path)
    print(f"copied best weights to: {export_path}")


if __name__ == "__main__":
    main()
