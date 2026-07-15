import numpy as np

REPLACEMENTS = {

    # Common mojibake
    "‚Äì": "–",
    "‚Äî": "—",
    "‚Äò": "'",
    "‚Äô": "'",
    "‚Äú": '"',
    "‚Äù": '"',
    "‚Ä¶": "...",
    "‚Ä¢": "•",
    "‚Ñ¢": "™",

    # Alternative mojibake
    "â€“": "–",
    "â€”": "—",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€¦": "...",
    "â€¢": "•",
    "â„¢": "™",
    "Â£": "£",

    # HTML entities
    "&quot;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&#39;": "'",

    # Extra encoding artefacts
    "\xa0": " ",
    "Â": "",
}

TEXT_TO_REPLACE = {
    r"(?i)^\s*$": np.nan,
    r"(?i)^nil$": np.nan,
    r"(?i)^null$": np.nan,
    r"(?i)^none$": np.nan,
    r"(?i)^n/?a\.?$": np.nan,
    r"(?i)^n\.?a\.?$": np.nan,
    r"(?i)^abstract to follow$": np.nan,
    r"(?i)^abstracts are not currently available in gtr.*$": np.nan,
    r"(?i)^awaiting public project summary$": np.nan,
    r"(?i)^tbc$": np.nan,
    r"(?i)^no public description$": np.nan,
    r"(?i)^abstract\s*": "",
    r"(?i)^not available$": np.nan,
    r"(?i)^not provided$": np.nan,
    r"(?i)^not applicable$": np.nan,
}

TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_VALUES = {"false", "f", "no", "n", "0", "0.0"}