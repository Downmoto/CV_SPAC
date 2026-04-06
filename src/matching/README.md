# src/matching

resident plate matching against a CSV database.

## ResidentMatcher

library module, no CLI. used by the inference pipeline.

```python
from src.matching.matcher import ResidentMatcher

matcher = ResidentMatcher(
    csv_path="data/db/residents.csv",
    use_fuzzy_matching=True,
    fuzzy_match_threshold=90.0,
)

result = matcher.match("ABC1234")
print(result.decision)       # "Access Granted" or "Access Denied"
print(result.matched)         # True if plate was found
print(result.matched_plate)   # matched plate string from db
print(result.score)           # 100.0 for exact, fuzzy score otherwise
print(result.record)          # full db row dict if matched
```

### matching logic

1. normalizes input text (uppercase, strip non-alphanumeric)
2. exact match against `normalized_plate` column
3. if no exact match and fuzzy matching is enabled, finds the best rapidfuzz ratio above threshold
4. checks the `status` column, `active` grants access, `inactive` denies it

### resident db format

CSV with columns: `plate_number`, `resident_name`, `unit`, `vehicle_color`, `status`

- `status`: `active` (access granted) or `inactive` (access denied despite match)
- if `status` column is missing, all residents default to `active`
