from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.utils.text import normalize_plate_text


@dataclass
class MatchResult:
    decision: str
    matched: bool
    matched_plate: str | None
    score: float
    record: dict[str, Any] | None


class ResidentMatcher:
    def __init__(
        self,
        csv_path: str,
        use_fuzzy_matching: bool = True,
        fuzzy_match_threshold: int = 90,
    ) -> None:
        self.csv_path = csv_path
        self.use_fuzzy_matching = use_fuzzy_matching
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "plate_number" not in df.columns:
            raise ValueError("resident db must include a 'plate_number' column")
        if "status" not in df.columns:
            df["status"] = "active"
        df["normalized_plate"] = df["plate_number"].astype(str).map(normalize_plate_text)
        return df

    def match(self, plate_text: str) -> MatchResult:
        normalized = normalize_plate_text(plate_text)
        if not normalized:
            return MatchResult("Access Denied", False, None, 0.0, None)

        exact = self.df[self.df["normalized_plate"] == normalized]
        if not exact.empty:
            row = {str(k): v for k, v in exact.iloc[0].to_dict().items()}
            is_active = str(row.get("status", "active")).lower() == "active"
            decision = "Access Granted" if is_active else "Access Denied"
            return MatchResult(decision, True, str(row.get("plate_number")), 100.0, row)

        if not self.use_fuzzy_matching or self.df.empty:
            return MatchResult("Access Denied", False, None, 0.0, None)

        best_score = -1.0
        best_row = None
        for _, row in self.df.iterrows():
            score = float(fuzz.ratio(normalized, row["normalized_plate"]))
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None and best_score >= self.fuzzy_match_threshold:
            row_dict = {str(k): v for k, v in best_row.to_dict().items()}
            is_active = str(row_dict.get("status", "active")).lower() == "active"
            decision = "Access Granted" if is_active else "Access Denied"
            return MatchResult(decision, True, str(row_dict.get("plate_number")), best_score, row_dict)

        return MatchResult("Access Denied", False, None, best_score if best_score > 0 else 0.0, None)
