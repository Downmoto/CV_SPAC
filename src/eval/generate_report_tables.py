import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate markdown report tables from detector and pipeline metrics"
    )
    parser.add_argument(
        "--detector-results-csv",
        default="runs/detect/outputs/detector_train/results.csv",
        help="ultralytics training results csv",
    )
    parser.add_argument(
        "--evaluation-summary-json",
        default="outputs/metrics/evaluation_summary.json",
        help="pipeline evaluation summary json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/evaluation_tables.md",
        help="output markdown report path",
    )
    return parser.parse_args()


def to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def fmt(v: float, n: int = 4) -> str:
    return f"{v:.{n}f}"


def load_detector_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"detector results csv not found: {path}")

    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"detector results csv is empty: {path}")

    best = max(rows, key=lambda r: to_float(r.get("metrics/mAP50-95(B)", "0")))
    last = rows[-1]

    return {
        "epochs_recorded": len(rows),
        "best_epoch": int(to_float(best.get("epoch", "0"))),
        "best_precision": to_float(best.get("metrics/precision(B)", "0")),
        "best_recall": to_float(best.get("metrics/recall(B)", "0")),
        "best_map50": to_float(best.get("metrics/mAP50(B)", "0")),
        "best_map5095": to_float(best.get("metrics/mAP50-95(B)", "0")),
        "last_epoch": int(to_float(last.get("epoch", "0"))),
        "last_precision": to_float(last.get("metrics/precision(B)", "0")),
        "last_recall": to_float(last.get("metrics/recall(B)", "0")),
        "last_map50": to_float(last.get("metrics/mAP50(B)", "0")),
        "last_map5095": to_float(last.get("metrics/mAP50-95(B)", "0")),
    }


def load_pipeline_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"evaluation summary json not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    base = payload.get("base_metrics", {})
    gt = payload.get("ground_truth_metrics")

    return {
        "base": base,
        "ground_truth": gt,
        "inference_json": payload.get("inference_json", ""),
        "ground_truth_csv": payload.get("ground_truth_csv", ""),
    }


def build_markdown(det: dict, pipe: dict) -> str:
    lines: list[str] = []

    lines.append("# evaluation tables")
    lines.append("")

    lines.append("## detector metrics")
    lines.append("")
    lines.append("| snapshot | epoch | precision | recall | mAP50 | mAP50-95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| best (by mAP50-95) | {det['best_epoch']} | {fmt(det['best_precision'])} | {fmt(det['best_recall'])} | {fmt(det['best_map50'])} | {fmt(det['best_map5095'])} |"
    )
    lines.append(
        f"| last recorded | {det['last_epoch']} | {fmt(det['last_precision'])} | {fmt(det['last_recall'])} | {fmt(det['last_map50'])} | {fmt(det['last_map5095'])} |"
    )
    lines.append("")
    lines.append(f"epochs recorded: {det['epochs_recorded']}")
    lines.append("")

    b = pipe["base"]
    lines.append("## pipeline base metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| samples | {int(b.get('samples', 0))} |")
    lines.append(f"| detections | {int(b.get('detections', 0))} |")
    lines.append(f"| detection_rate | {fmt(float(b.get('detection_rate', 0.0)), 6)} |")
    lines.append(f"| ocr_nonempty | {int(b.get('ocr_nonempty', 0))} |")
    lines.append(f"| ocr_nonempty_rate | {fmt(float(b.get('ocr_nonempty_rate', 0.0)), 6)} |")
    lines.append(f"| pred_access_granted | {int(b.get('pred_access_granted', 0))} |")
    lines.append(f"| pred_access_denied | {int(b.get('pred_access_denied', 0))} |")
    lines.append(f"| mean_detector_conf | {fmt(float(b.get('mean_detector_conf', 0.0)), 6)} |")
    lines.append(f"| mean_ocr_conf | {fmt(float(b.get('mean_ocr_conf', 0.0)), 6)} |")
    lines.append("")

    gt = pipe.get("ground_truth")
    if gt and isinstance(gt, dict) and gt.get("decision_metrics"):
        dm = gt["decision_metrics"]
        cm = dm.get("confusion", {})
        pm = gt.get("plate_metrics", {})

        lines.append("## decision metrics (labeled)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| labeled_samples | {int(gt.get('labeled_samples', 0))} |")
        lines.append(f"| unlabeled_samples | {int(gt.get('unlabeled_samples', 0))} |")
        lines.append(f"| positive_label | {dm.get('positive_label', '')} |")
        lines.append(f"| accuracy | {fmt(float(dm.get('accuracy', 0.0)), 6)} |")
        lines.append(f"| precision | {fmt(float(dm.get('precision', 0.0)), 6)} |")
        lines.append(f"| recall | {fmt(float(dm.get('recall', 0.0)), 6)} |")
        lines.append(f"| f1 | {fmt(float(dm.get('f1', 0.0)), 6)} |")
        lines.append("")
        lines.append("| confusion term | count |")
        lines.append("|---|---:|")
        lines.append(f"| tp | {int(cm.get('tp', 0))} |")
        lines.append(f"| tn | {int(cm.get('tn', 0))} |")
        lines.append(f"| fp | {int(cm.get('fp', 0))} |")
        lines.append(f"| fn | {int(cm.get('fn', 0))} |")
        lines.append("")
        lines.append("| plate metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| plate_labeled_samples | {int(pm.get('labeled_samples', 0))} |")
        lines.append(
            f"| plate_exact_match_rate | {fmt(float(pm.get('exact_match_rate', 0.0)), 6)} |"
        )
        lines.append("")
    else:
        lines.append("## decision metrics (labeled)")
        lines.append("")
        lines.append("ground-truth evaluation is not available yet in the summary json.")
        lines.append("")

    lines.append("## artifact paths")
    lines.append("")
    lines.append(f"- inference json: {pipe.get('inference_json', '')}")
    if pipe.get("ground_truth_csv"):
        lines.append(f"- ground truth csv: {pipe.get('ground_truth_csv', '')}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    det = load_detector_summary(Path(args.detector_results_csv))
    pipe = load_pipeline_summary(Path(args.evaluation_summary_json))

    md = build_markdown(det, pipe)

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"saved markdown tables to: {out}")


if __name__ == "__main__":
    main()
