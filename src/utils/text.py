import re


def normalize_plate_text(text: str) -> str:
    """Normalize plate text for matching.

    Rules:
    - uppercase
    - remove non-alphanumeric characters
    """
    if text is None:
        return ""
    text = text.upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)
