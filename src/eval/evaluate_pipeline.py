import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.text import normalize_plate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate spac inference outputs")
    parser.add_argument(
        "--inference-json",
        default="outputs/predictions/inference_results.json",
        help="pipeline output json path",
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=None,
        help="optional csv with columns: image_path, expected_decision, expected_plate(optional)",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/metrics/evaluation_summary.json",
        help="path to write evaluation summary json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/metrics/evaluation_rows.csv",
        help="path to write merged per-image rows when ground truth is provided",
    )
    parser.add_argument(
        "--decision-positive-label",
        default="Access Granted",
        help="positive decision class for precision/recall/f1",
    )
    return parser.parse_args()


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def load_inference_df(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    normalized_rows: list[dict] = []
    for row in rows:
        image_path = str(row.get("image_path", ""))
        image_name = Path(image_path).name
        bbox = row.get("detector", {}).get("bbox_xyxy")
        det_conf = float(row.get("detector", {}).get("conf", 0.0) or 0.0)
        ocr_text = str(row.get("ocr", {}).get("plate_text", "") or "")
        ocr_conf = float(row.get("ocr", {}).get("confidence", 0.0) or 0.0)
        decision = str(row.get("decision", {}).get("label", ""))
        matched = bool(row.get("decision", {}).get("matched", False))
        match_score = float(row.get("decision", {}).get("match_score", 0.0) or 0.0)

        normalized_rows.append(
            {
                "image_path": image_path,
                "image_name": image_name,
                "detected": bbox is not None,
                "detector_conf": det_conf,
                "ocr_text": ocr_text,
                "ocr_text_norm": normalize_plate_text(ocr_text),
                "ocr_nonempty": bool(ocr_text.strip()),
                "ocr_conf": ocr_conf,
                "pred_decision": decision,
                "pred_matched": matched,
                "pred_match_score": match_score,
            }
        )
    return pd.DataFrame(normalized_rows)


def compute_base_metrics(df: pd.DataFrame) -> dict:
    samples = int(len(df))
    detections = int(df["detected"].sum()) if samples else 0
    ocr_nonempty = int(df["ocr_nonempty"].sum()) if samples else 0
    granted = int((df["pred_decision"] == "Access Granted").sum()) if samples else 0
    denied = int((df["pred_decision"] == "Access Denied").sum()) if samples else 0

    return {
        "samples": samples,
        "detections": detections,
        "detection_rate": round(safe_div(detections, samples), 6),
        "ocr_nonempty": ocr_nonempty,
        "ocr_nonempty_rate": round(safe_div(ocr_nonempty, samples), 6),
        "pred_access_granted": granted,
        "pred_access_denied": denied,
        "mean_detector_conf": round(float(df["detector_conf"].mean()) if samples else 0.0, 6),
        "mean_ocr_conf": round(float(df["ocr_conf"].mean()) if samples else 0.0, 6),
    }


def evaluate_with_ground_truth(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    positive_label: str,
) -> tuple[dict, pd.DataFrame]:
    gt = gt_df.copy()
    if "image_path" not in gt.columns or "expected_decision" not in gt.columns:
        raise ValueError("ground truth csv must include image_path and expected_decision columns")

    gt["image_path"] = gt["image_path"].fillna("").astype(str)
    gt["image_name"] = gt["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]

    if "expected_plate" not in gt.columns:
        gt["expected_plate"] = ""
    gt["expected_plate"] = gt["expected_plate"].fillna("").astype(str)
    gt["expected_plate_norm"] = gt["expected_plate"].map(normalize_plate_text)

    merged = pred_df.merge(gt, on="image_name", how="left", suffixes=("", "_gt"))

    labeled = merged[merged["expected_decision"].notna()].copy()
    unlabeled_count = int((merged["expected_decision"].isna()).sum())

    if labeled.empty:
        return {
            "labeled_samples": 0,
            "unlabeled_samples": unlabeled_count,
            "message": "no labeled rows matched by image filename",
        }, merged

    y_true = labeled["expected_decision"].astype(str)
    y_pred = labeled["pred_decision"].astype(str)

    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    tn = int(((y_true != positive_label) & (y_pred != positive_label)).sum())
    fp = int(((y_true != positive_label) & (y_pred == positive_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred != positive_label)).sum())

    accuracy = safe_div(tp + tn, len(labeled))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    exp_series = labeled["expected_plate_norm"].astype(str).str.upper()
    has_expected_plate = (exp_series.str.len() > 0) & (exp_series != "UNKNOWN")
    plate_cmp = labeled[has_expected_plate].copy()
    unknown_or_blank = int((~has_expected_plate).sum())

    if plate_cmp.empty:
        plate_exact_rate = 0.0
        plate_labeled_count = 0
    else:
        plate_exact = (
            plate_cmp["expected_plate_norm"].astype(str)
            == plate_cmp["ocr_text_norm"].astype(str)
        )
        plate_labeled_count = int(len(plate_cmp))
        plate_exact_rate = float(plate_exact.mean())

    summary = {
        "labeled_samples": int(len(labeled)),
        "unlabeled_samples": unlabeled_count,
        "decision_metrics": {
            "positive_label": positive_label,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "confusion": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
        },
        "plate_metrics": {
            "labeled_samples": plate_labeled_count,
            "excluded_unknown_or_blank": unknown_or_blank,
            "exact_match_rate": round(plate_exact_rate, 6),
        },
    }
    return summary, merged


def main() -> None:
    args = parse_args()
    inference_json = Path(args.inference_json)
    if not inference_json.exists():
        raise FileNotFoundError(f"inference json not found: {inference_json}")

    pred_df = load_inference_df(inference_json)
    summary = {
        "inference_json": str(inference_json),
        "base_metrics": compute_base_metrics(pred_df),
    }

    merged_df = None
    if args.ground_truth_csv:
        gt_path = Path(args.ground_truth_csv)
        if not gt_path.exists():
            raise FileNotFoundError(f"ground truth csv not found: {gt_path}")
        gt_df = pd.read_csv(gt_path)
        gt_summary, merged_df = evaluate_with_ground_truth(
            pred_df,
            gt_df,
            positive_label=args.decision_positive_label,
        )
        summary["ground_truth_csv"] = str(gt_path)
        summary["ground_truth_metrics"] = gt_summary

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if merged_df is not None:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(out_csv, index=False)
        print(f"saved merged evaluation rows to: {out_csv}")

    print(f"saved evaluation summary to: {out_json}")


if __name__ == "__main__":
    main()
