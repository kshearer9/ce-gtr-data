from pathlib import Path
import pandas as pd
from utils.cleaning import (normalise_name, convert_to_string, 
                            convert_to_date, clean_text_columns, 
                            convert_to_category, convert_to_numeric)

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed" / "openalex" 
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


STRING_COLUMNS = [
    "project_id",
    "project_title",
    "grant_reference",
    "project_openalex_url",
    "authors",
    "institutions",
    "topics",
    "doi",
    "url",
    "openalex_url"
]

TEXT_COLUMNS = [
    "title",
    "abstract"
]

NUMERIC_COLUMNS = [
    "cited_by",
    "fwci"
]

DATE_COLUMNS = [
    "publication_date"
]

CATEGORY_COLUMNS = [
    "type",
    "domain",
    "field",
    "subfield"
]

def clean_authors(authors):
    """
    Normalise semicolon-separated author names.
    """
    if pd.isna(authors):
        return pd.NA
    cleaned = []
    for name in str(authors).split(";"):
        name = name.strip()
        if not name:
            continue
        normalised = normalise_name(name)
        if normalised:
            cleaned.append(normalised)
    return "; ".join(cleaned) if cleaned else pd.NA


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    processed_file = INPUT_DIR / "openalex_outcomes_latest.csv"
    if processed_file.exists():
        input_file = processed_file
    else:
        raise FileNotFoundError(
            "Could not find openalex_outcomes_latest.csv")
    
    df = pd.read_csv(input_file, encoding="utf-8")
    df = clean_text_columns(df, *TEXT_COLUMNS)
    df = convert_to_numeric(df, *NUMERIC_COLUMNS)
    df = convert_to_date(df, *DATE_COLUMNS)
    df = convert_to_category(df, *CATEGORY_COLUMNS)
    df = convert_to_string(df, *STRING_COLUMNS)
    df["authors_clean"] = df["authors"].apply(clean_authors)
    output_file = OUTPUT_DIR / "openalex_all_outcomes_clean.csv"
    df.to_csv(output_file, index = False, encoding = "utf-8")
    print(f"Saved: {output_file.name} "
    f"({len(df):,} rows × {len(df.columns)} columns)\n")    


if __name__ == "__main__":
    main()