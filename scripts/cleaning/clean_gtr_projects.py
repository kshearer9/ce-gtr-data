import pandas as pd
import html
import re
from pathlib import Path
import numpy as np
import unicodedata
from utils.constants import REPLACEMENTS
from utils.cleaning import (normalise_name, convert_to_string, convert_to_date,
                            clean_text_columns, convert_to_category, 
                            convert_to_numeric)

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed" / "gtr"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# COLUMNS TO CLEAN
# ---------------------------------------------------------------------------

TEXT_COLS = ["title", 
             "abstract_text", 
             "tech_abstract_text", 
             "potential_impact"]

STRING_COLS = ["project_id",
               "lead_organisation",
               "participant_organisations",
               "principal_investigator",
               "sectors",
               "fund_start",
               "fund_end",
               "grant_reference",
               "discipline_primary",
               "research_subjects",
               "research_topics"
               "gtr_url"]

CATEGORY_COLS = ["lead_funder",
                 "status",
                 "grant_category",
                 "discipline_source"]

NUMERIC_COLS = ["value_pounds"]

COLS_TO_DROP = ["funding_data_available",
                "n_research_subjects",
                "matched_search_term",
                "filter_decision",
                "tier1_matches",
                "tier2_matches",
                "tier3_matches"]


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

# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    processed_file = INPUT_DIR / "gtr_projects_latest.csv"
    if processed_file.exists():
        input_file = processed_file
    else:
        raise FileNotFoundError(
            "Could not find gtr_projects_latest.csv")
            
    df = pd.read_csv(input_file, encoding="utf-8")
    df = clean_df(df)
    df = clean_text_columns(df, *TEXT_COLS)
    df = convert_to_numeric(df, *NUMERIC_COLS)
    df = convert_to_category(df, *CATEGORY_COLS)
    df = convert_to_string(df, *STRING_COLS)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    if "principal_investigator" in df.columns:
        df["pi_clean"] = df["principal_investigator"].apply(normalise_name)

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
        print("Duplicate project id :", df["project_id"].duplicated().sum())
    if "title" in df.columns:
        print("Duplicate title :", df["title"].duplicated().sum())

    output_file = OUTPUT_DIR / "gtr_projects_clean.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print("\n" + "=" * 40)
    print("GTR data cleaning completed.")
    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Saved          : {output_file.name}")
    print("=" * 40)


if __name__ == "__main__":
    main()