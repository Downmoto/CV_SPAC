import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spac workflow runner")
    parser.add_argument("--config", default="configs/default.yaml", help="config yaml path")

    parser.add_argument("--download", action="store_true", help="download and unzip kaggle dataset")
    parser.add_argument("--prepare", action="store_true", help="prepare kaggle dataset")
    parser.add_argument("--seed-db", action="store_true", help="create sample resident db")
    parser.add_argument("--train", action="store_true", help="train detector")
    parser.add_argument("--extract-crops", action="store_true", help="extract plate crops for ocr training")
    parser.add_argument("--train-ocr", action="store_true", help="train crnn plate ocr model")
    parser.add_argument("--infer", action="store_true", help="run end-to-end inference")
    parser.add_argument("--eval", action="store_true", help="run evaluation summary")
    parser.add_argument("--report", action="store_true", help="generate markdown report tables")
    parser.add_argument("--all", action="store_true", help="run download -> prepare -> train -> infer -> eval -> report")
    parser.add_argument("--force-download", action="store_true", help="force kaggle download even if zip exists")
    parser.add_argument("--ocr-epochs", type=int, default=None, help="override crnn training epochs")
    parser.add_argument("--ocr-repeat-factor", type=int, default=None, help="override crnn dataset repeat factor")

    parser.add_argument("--image", default=None, help="override single inference image")
    parser.add_argument("--image-dir", default=None, help="override inference image directory")
    parser.add_argument("--ground-truth-csv", default=None, help="override ground-truth csv path")
    parser.add_argument(
        "--use-ground-truth",
        action="store_true",
        help="include ground-truth evaluation if csv exists",
    )
    return parser.parse_args()


def run_download(cfg: dict[str, Any], force_download: bool) -> None:
    ds = cfg.get("dataset", {})
    source = cfg.get("source", {})

    dataset_slug = str(source.get("kaggle_dataset", "andrewmvd/car-plate-detection"))
    download_dir = str(ds.get("download_dir", "data/raw/car-plate-detection/"))

    kaggle_exe = Path(sys.executable).parent / "kaggle"
    kaggle_cmd = str(kaggle_exe) if kaggle_exe.exists() else "kaggle"

    cmd = [
        kaggle_cmd,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        download_dir,
        "--unzip",
    ]
    if force_download:
        cmd.append("--force")

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_prepare(cfg: dict[str, Any]) -> None:
    ds = cfg.get("dataset", {})
    args = [
        "--raw-dir",
        str(ds.get("raw_dir", "data/raw/car-plate-detection")),
        "--out-dir",
        str(ds.get("out_dir", "data/processed/car_plate_kaggle")),
        "--seed",
        str(ds.get("seed", 42)),
        "--train-ratio",
        str(ds.get("train_ratio", 0.8)),
        "--val-ratio",
        str(ds.get("val_ratio", 0.1)),
        "--test-ratio",
        str(ds.get("test_ratio", 0.1)),
        "--plate-class-name",
        str(ds.get("plate_class_name", "plate")),
    ]
    if bool(ds.get("clear_out_dir", False)):
        args.append("--clear-out-dir")

    run_module("src.data.prepare_kaggle_car_plate_dataset", args)


def run_seed_db(cfg: dict[str, Any], ground_truth_csv_override: str | None = None) -> None:
    paths = cfg.get("paths", {})
    ds = cfg.get("dataset", {})
    db_seed = cfg.get("db_seed", {})

    gt_csv = Path(
        ground_truth_csv_override
        or str(paths.get("ground_truth_csv", "outputs/metrics/ground_truth_template.csv"))
    )
    inference_json = Path(str(paths.get("output_json", "outputs/predictions/inference_results.json")))

    if not gt_csv.exists():
        if inference_json.exists():
            print(
                "ground-truth csv not found for seed-db; "
                f"auto-generating template at {gt_csv} from {inference_json}"
            )
            run_module(
                "src.eval.create_ground_truth_template",
                [
                    "--inference-json",
                    str(inference_json),
                    "--output-csv",
                    str(gt_csv),
                ],
            )
        else:
            raise FileNotFoundError(
                "seed-db requires a ground-truth csv with expected_plate values. "
                f"missing: {gt_csv}. also could not auto-generate because inference json is missing: {inference_json}. "
                "run --infer first, or provide --ground-truth-csv to an existing file."
            )

    args = [
        "--ground-truth-csv",
        str(gt_csv),
        "--inference-json",
        str(inference_json),
        "--fallback-to-ocr",
        "--output-csv",
        str(paths.get("resident_db_csv", "data/db/residents.csv")),
        "--seed",
        str(db_seed.get("seed", ds.get("seed", 42))),
        "--active-ratio",
        str(db_seed.get("active_ratio", 0.85)),
    ]
    run_module("src.data.create_sample_db", args)


def run_train(cfg: dict[str, Any]) -> None:
    tr = cfg.get("training", {})
    args = [
        "--data-yaml",
        str(tr.get("data_yaml", "data/processed/car_plate_kaggle/dataset.yaml")),
        "--model",
        str(tr.get("model", "yolov8n.pt")),
        "--epochs",
        str(tr.get("epochs", 80)),
        "--imgsz",
        str(tr.get("imgsz", 640)),
        "--batch",
        str(tr.get("batch", 16)),
        "--project",
        str(tr.get("project", "outputs")),
        "--name",
        str(tr.get("name", "detector_train")),
    ]
    device = tr.get("device")
    if device is not None:
        args.extend(["--device", str(device)])
    run_module("src.train.train_detector", args)



def run_extract_crops(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    ds = cfg.get("dataset", {})
    gt_csv = str(paths.get("ground_truth_csv", "outputs/metrics/ground_truth_template.csv"))
    out_dir = str(ds.get("out_dir", "data/processed/car_plate_kaggle"))

    args = ["--config", "configs/default.yaml", "--output-dir", "data/plate_crops"]
    if Path(gt_csv).exists():
        args.extend(["--ground-truth-csv", gt_csv])
    run_module("src.data.extract_plate_crops", args)


def run_train_ocr(cfg: dict[str, Any], epochs_override: int | None = None, repeat_factor_override: int | None = None) -> None:
    ocr_cfg = cfg.get("ocr", {})
    crnn_weights = str(ocr_cfg.get("crnn_weights", "models/plate_crnn.pt"))

    args = [
        "--crops-dir", "data/plate_crops/images",
        "--labels-csv", "data/plate_crops/labels.csv",
        "--output-weights", crnn_weights,
    ]
    if epochs_override is not None:
        args.extend(["--epochs", str(epochs_override)])
    if repeat_factor_override is not None:
        args.extend(["--repeat-factor", str(repeat_factor_override)])
    run_module("src.train.train_plate_ocr", args)


def run_infer(cfg_path: str, image: str | None, image_dir: str | None) -> None:
    args = ["--config", cfg_path]
    if image:
        args.extend(["--image", image])
    elif image_dir:
        args.extend(["--image-dir", image_dir])
    else:
        cfg = load_config(cfg_path)
        default_image_dir = str(
            cfg.get("dataset", {}).get("out_dir", "data/processed/car_plate_kaggle")
        )
        args.extend(["--image-dir", str(Path(default_image_dir) / "images" / "test")])

    run_module("src.infer.run_inference", args)



def run_eval(
    cfg: dict[str, Any],
    ground_truth_csv_override: str | None,
    use_ground_truth: bool,
) -> None:
    paths = cfg.get("paths", {})
    ev = cfg.get("evaluation", {})

    inference_json = str(paths.get("output_json", "outputs/predictions/inference_results.json"))
    output_json = str(paths.get("evaluation_summary_json", "outputs/metrics/evaluation_summary.json"))
    output_csv = str(paths.get("evaluation_rows_csv", "outputs/metrics/evaluation_rows.csv"))
    decision_positive_label = str(ev.get("decision_positive_label", "Access Granted"))

    args = [
        "--inference-json",
        inference_json,
        "--output-json",
        output_json,
        "--output-csv",
        output_csv,
        "--decision-positive-label",
        decision_positive_label,
    ]

    gt_csv = ground_truth_csv_override or str(paths.get("ground_truth_csv", ""))
    if use_ground_truth and gt_csv and Path(gt_csv).exists():
        args.extend(["--ground-truth-csv", gt_csv])
    elif use_ground_truth and gt_csv and not Path(gt_csv).exists():
        print(f"ground-truth csv not found, running base evaluation only: {gt_csv}")

    run_module("src.eval.evaluate_pipeline", args)



def run_report(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    detector_results_csv = str(
        paths.get("detector_results_csv", "runs/detect/outputs/detector_train/results.csv")
    )
    evaluation_summary_json = str(
        paths.get("evaluation_summary_json", "outputs/metrics/evaluation_summary.json")
    )
    output_md = str(paths.get("report_tables_md", "docs/evaluation_tables.md"))

    args = [
        "--detector-results-csv",
        detector_results_csv,
        "--evaluation-summary-json",
        evaluation_summary_json,
        "--output-md",
        output_md,
    ]
    run_module("src.eval.generate_report_tables", args)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.all:
        run_download(cfg, args.force_download)
        run_prepare(cfg)
        run_train(cfg)
        run_infer(args.config, args.image, args.image_dir)
        run_eval(cfg, args.ground_truth_csv, args.use_ground_truth)
        run_report(cfg)
        return

    if args.download:
        run_download(cfg, args.force_download)
    if args.seed_db:
        run_seed_db(cfg, args.ground_truth_csv)
    if args.prepare:
        run_prepare(cfg)
    if args.train:
        run_train(cfg)
    if args.extract_crops:
        run_extract_crops(cfg)
    if args.train_ocr:
        run_train_ocr(cfg, args.ocr_epochs, args.ocr_repeat_factor)
    if args.infer:
        run_infer(args.config, args.image, args.image_dir)
    if args.eval:
        run_eval(cfg, args.ground_truth_csv, args.use_ground_truth)
    if args.report:
        run_report(cfg)

    if not any([args.download, args.seed_db, args.prepare, args.train, args.extract_crops, args.train_ocr, args.infer, args.eval, args.report, args.all]):
        raise ValueError("no action provided. use --download/--prepare/--train/--extract-crops/--train-ocr/--infer/--eval/--report/--all")


if __name__ == "__main__":
    main()
