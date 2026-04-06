from pathlib import Path

import pandas as pd


def main() -> None:
    db_path = Path("data/db/residents.csv")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "plate_number": "ABC123",
            "resident_name": "John Doe",
            "unit": "A-101",
            "vehicle_color": "white",
            "status": "active",
        },
        {
            "plate_number": "XYZ789",
            "resident_name": "Sara Lee",
            "unit": "B-204",
            "vehicle_color": "black",
            "status": "active",
        },
        {
            "plate_number": "TEST000",
            "resident_name": "Old Tenant",
            "unit": "C-010",
            "vehicle_color": "blue",
            "status": "inactive",
        },
    ]

    pd.DataFrame(rows).to_csv(db_path, index=False)
    print(f"created sample resident db at: {db_path}")


if __name__ == "__main__":
    main()
