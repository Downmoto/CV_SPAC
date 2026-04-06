import argparse

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
    parser.add_argument("--project", default="outputs")
    parser.add_argument("--name", default="detector_train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
