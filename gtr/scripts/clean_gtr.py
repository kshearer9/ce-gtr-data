import pandas as pd
import html
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GTR_DIR = SCRIPT_DIR.parent
PROC_DIR = GTR_DIR / "data" / "processed"
CLEAN_DIR = GTR_DIR / "data" / "cleaned"

for d in (PROC_DIR, CLEAN_DIR):
    d.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROC_DIR / "gtr_ce_projects_latest.csv"
OUTPUT_FILE = CLEAN_DIR / "gtr_ce_projects_clean.csv"

# ---------------------------------------------------------------------------
# ENCODING / HTML FIXES
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# COLUMNS TO CLEAN
# ---------------------------------------------------------------------------

TEXT_COLUMNS = ["title", "abstract_text", 
                "tech_abstract_text", "potential_impact"]

# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------

def clean_text(value):
    """Cleans raw text by decoding HTML entities, removing HTM tags, removing
    markdown formatting, fixing encoding issues and normalising whitespace."""
    if pd.isna(value):
        return value, False
    text = str(value)
    original = text
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove Markdown bold markers
    text = text.replace("**", "")
    # Fix encoding artefacts
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    # Normalise whitespace
    text = " ".join(text.split())
    return text, text != original


def normalise_title(text):
    """Creates a simplified version of a title for comparison purposes."""
    if pd.isna(text):
        return text
    text = text.lower()
    text = re.sub(r"[–—-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)
    modified_cells = 0
    for col in TEXT_COLUMNS:
        cleaned_values = []
        for value in df[col]:
            cleaned, changed = clean_text(value)
            cleaned_values.append(cleaned)
            if changed:
                modified_cells += 1
        df[col] = cleaned_values

    # Missing value reporting
    print("\nMissing Values")
    print("-" * 40)
    missing = df.isna().sum()
    for col, value in missing.items():
        if value > 0:
            print(f"{col:<30}{value}")

    # Duplicate checks and cleaning
    print("\nDuplicate Checks")
    print("-" * 40)
    if "project_id" in df.columns:
        id_duplicates = df["project_id"].duplicated().sum()
        print("Duplicate project_id :", id_duplicates)
        # Remove projects with duplicate project id
        if id_duplicates > 0:
            df = df.drop_duplicates(subset="project_id", keep="first")
            print("Removed projects with duplicate project ID's.")
    if "title" in df.columns:
        print("Duplicate title :", df["title"].duplicated().sum())
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(normalise_title)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 40)
    print("GTR data cleaning completed.")
    print(f"Modified cells : {modified_cells}")
    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Saved          : {OUTPUT_FILE}")
    print("=" * 40)


if __name__ == "__main__":
    main()