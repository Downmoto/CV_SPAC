import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="create a ground-truth template csv from inference results"
    )
    parser.add_argument(
        "--inference-json",
        default="outputs/predictions/inference_results.json",
        help="pipeline output json path",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/metrics/ground_truth_template.csv",
        help="where to write template csv",
    )
    parser.add_argument(
        "--default-decision",
        default="Access Denied",
        help="initial expected_decision value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_path = Path(args.inference_json)
    if not inference_path.exists():
        raise FileNotFoundError(f"inference json not found: {inference_path}")

    df = pd.read_json(inference_path)
    out = pd.DataFrame(
        {
            "image_path": df["image_path"].astype(str),
            "expected_decision": args.default_decision,
            "expected_plate": "",
            "notes": "",
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"saved ground-truth template to: {output_csv}")


if __name__ == "__main__":
    main()
