import argparse
import random
from pathlib import Path

import pandas as pd

from src.utils.text import normalize_plate_text


FIRST_NAMES = [
    "Ava",
    "Noah",
    "Liam",
    "Mia",
    "Ethan",
    "Olivia",
    "Aria",
    "Lucas",
    "Emma",
    "Mason",
    "Sophia",
    "Isla",
    "Leo",
    "Nora",
    "Ryan",
    "Zara",
]

LAST_NAMES = [
    "Shah",
    "Patel",
    "Khan",
    "Singh",
    "Garcia",
    "Brown",
    "Kim",
    "Ahmed",
    "Miller",
    "Wilson",
    "Davis",
    "Taylor",
    "Ali",
    "Clark",
    "Thomas",
    "Lee",
]

VEHICLE_COLORS = [
    "white",
    "black",
    "silver",
    "gray",
    "blue",
    "red",
    "green",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="create a random resident db from expected plates in a ground-truth csv"
    )
    parser.add_argument(
        "--ground-truth-csv",
        default="outputs/metrics/ground_truth_template.csv",
        help="path to ground-truth csv that contains expected_plate column",
    )
    parser.add_argument(
        "--output-csv",
        default="data/db/residents.csv",
        help="where to write the generated resident db",
    )
    parser.add_argument(
        "--inference-json",
        default="outputs/predictions/inference_results.json",
        help="inference json used as fallback source of plates when expected_plate is empty",
    )
    parser.add_argument(
        "--fallback-to-ocr",
        action="store_true",
        help="fallback to ocr_text from inference json when expected_plate values are missing",
    )
    parser.add_argument("--seed", type=int, default=42, help="rng seed")
    parser.add_argument(
        "--active-ratio",
        type=float,
        default=0.85,
        help="share of residents marked active (0.0 to 1.0)",
    )
    return parser.parse_args()


def _random_unit(rng: random.Random) -> str:
    tower = rng.choice(["A", "B", "C", "D"])
    floor = rng.randint(1, 20)
    door = rng.randint(1, 12)
    return f"{tower}-{floor:02d}{door:02d}"


def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _load_unique_valid_plates(
    ground_truth_csv: Path,
    inference_json: Path,
    fallback_to_ocr: bool,
) -> list[str]:
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"ground-truth csv not found: {ground_truth_csv}")

    df = pd.read_csv(ground_truth_csv)
    if "expected_plate" not in df.columns:
        raise ValueError("ground-truth csv must contain expected_plate column")

    plates = (
        df["expected_plate"]
        .fillna("")
        .astype(str)
        .map(normalize_plate_text)
    )
    plates = plates[(plates != "") & (plates != "UNKNOWN")]
    unique_plates = sorted(set(plates.tolist()))

    if unique_plates:
        return unique_plates

    if fallback_to_ocr and inference_json.exists():
        inf_df = pd.read_json(inference_json)

        ocr_series = None
        if "ocr_text" in inf_df.columns:
            ocr_series = inf_df["ocr_text"]
        elif "ocr" in inf_df.columns:
            ocr_series = inf_df["ocr"].map(
                lambda item: item.get("plate_text", "")
                if isinstance(item, dict)
                else ""
            )

        if ocr_series is not None:
            pred_plates = ocr_series.fillna("").astype(str).map(normalize_plate_text)
            pred_plates = pred_plates[(pred_plates != "") & (pred_plates != "UNKNOWN")]
            unique_pred_plates = sorted(set(pred_plates.tolist()))
            if unique_pred_plates:
                print(
                    "warning: no valid expected_plate values found; "
                    f"falling back to ocr text from inference json: {inference_json}"
                )
                return unique_pred_plates

    raise ValueError(
        "no valid expected_plate values found in ground-truth csv. "
        "fill expected_plate values first, or enable fallback with --fallback-to-ocr and provide inference json."
    )


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.active_ratio <= 1.0):
        raise ValueError("--active-ratio must be between 0.0 and 1.0")

    gt_path = Path(args.ground_truth_csv)
    inference_path = Path(args.inference_json)
    db_path = Path(args.output_csv)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    unique_plates = _load_unique_valid_plates(
        ground_truth_csv=gt_path,
        inference_json=inference_path,
        fallback_to_ocr=args.fallback_to_ocr,
    )

    rows = []
    for plate in unique_plates:
        rows.append(
            {
                "plate_number": plate,
                "resident_name": _random_name(rng),
                "unit": _random_unit(rng),
                "vehicle_color": rng.choice(VEHICLE_COLORS),
                "status": "active" if rng.random() <= args.active_ratio else "inactive",
            }
        )

    pd.DataFrame(rows).to_csv(db_path, index=False)
    print(f"created resident db at: {db_path}")
    print(f"source ground-truth csv: {gt_path}")
    print(f"rows created: {len(rows)}")


if __name__ == "__main__":
    main()
