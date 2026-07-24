from pathlib import Path
import pandas as pd
from utils.constants import TEXT_TO_REPLACE
from utils.cleaning import (normalise_name, convert_to_string, 
                            convert_to_date, clean_text_columns, 
                            convert_to_category, convert_to_numeric)

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed" / "openalex" 
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


STRING_COLS = [
    "project_id",
    "project_title",
    "grant_reference",
    "project_openalex_url",
    "ukri_url"
]

TEXT_COLS = ["description"]

NUMERIC_COLS = [
    "funding_amount",
    "primary_topic_score"
]

DATE_COLS = [
    "start_date",
    "end_date"
]

CATEGORY_COLS = [
    "currency",
    "funding_type",
    "primary_topic",
    "domain",
    "field",
    "subfield"
]

def clean_df(df):
    # Remove duplicate outcomes
    if "project_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("project_id")
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} duplicate outcomes")
    # Replace missing abstracts with nan
    df = df.replace(TEXT_TO_REPLACE, regex=True)
    df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df

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
    processed_file = INPUT_DIR / "openalex_projects_latest.csv"
    if processed_file.exists():
        input_file = processed_file
    else:
        raise FileNotFoundError(
            "Could not find openalex_projects_latest.csv")
    
    df = pd.read_csv(input_file, encoding="utf-8")
    df = clean_df(df)
    df = clean_text_columns(df, *TEXT_COLS)
    df = convert_to_numeric(df, *NUMERIC_COLS)
    df = convert_to_date(df, *DATE_COLS)
    df = convert_to_string(df, *STRING_COLS)
    df["funding_type"] = df["funding_type"].str.replace("_", " ")
    df = convert_to_category(df, *CATEGORY_COLS)
    output_file = OUTPUT_DIR / "openalex_projects_clean.csv"
    df.to_csv(output_file, index = False, encoding = "utf-8")
    print(f"Saved: {output_file.name} "
    f"({len(df):,} rows × {len(df.columns)} columns)\n") 


if __name__ == "__main__":
    main()