"""
Clean processed Scopus outcome datasets.

The script:
    1. Cleans outcome metadata extracted from the Scopus API.
    2. Standardises data types and text fields.
    3. Removes duplicate outcomes.
    4. Produces cleaned outcome, institution and reference datasets.

Exported outputs:
    - scopus_all_outcomes_clean.csv – Cleaned outcome metadata.
    - scopus_institutions_clean.csv – Cleaned institution affiliation data.
    - scopus_references_clean.csv – Cleaned cited reference data.
"""

from pathlib import Path
import pandas as pd
from utils.cleaning import (normalise_name, convert_to_string, 
                            convert_to_date, clean_text_columns, 
                            convert_to_category, convert_to_numeric)
from utils.constants import TEXT_TO_REPLACE

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed" / "scopus" 
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLEANING CONFIG
# ---------------------------------------------------------------------------

STRING_COLUMNS = [
    "project_id",
    "project_title",
    "grant_reference",
    "doi",
    "eid",
    "scopus_id",
    "journal",
    "volume",
    "issue",
    "start_page",
    "end_page",
    "page_range",
    "scopus_url",
    "institutions",
    "issn",
    "authors",
    "subject_areas",
    "keywords",
    "journal"
]

TEXT_COLUMNS = [
    "title",
    "abstract",
]

NUMERIC_COLUMNS = [
    "citation_count",
    "source_id",
    "pubmed_id",
    "reference_count"
]

DATE_COLUMNS = [
    "publication_date"
]

CATEGORY_COLUMNS = [
    "publication_type",
    "aggregation_type"
]

COLS_TO_DROP = [
    "open_access",
    "open_access_flag",
    "publisher",
    "page_range"
]


# ---------------------------------------------------------------------------
# CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

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

def clean_df(df):
    removed_dupes = pd.DataFrame
    # Remove duplicate project-outcome matches
    if {"project_id", "scopus_id"}.issubset(df.columns):
        before = len(df)
        # Keep the duplicates that will be removed
        removed_dupes = df[df.duplicated(subset=["project_id", "scopus_id"], keep="first")]
        # Keep only the first occurrence
        df = df.drop_duplicates(subset=["project_id", "scopus_id"])
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} duplicate outcomes")
    # Replace placeholder text with NaN
    df = df.replace(TEXT_TO_REPLACE, regex=True)
    # Remove leading/trailing whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df, removed_dupes


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    processed_file = INPUT_DIR / "scopus_outcomes_latest.csv"
    if processed_file.exists():
        input_file = processed_file
    else:
        raise FileNotFoundError(
            "Could not find scopus_outcomes_latest.csv")
    
    df = pd.read_csv(input_file, encoding="utf-8")
    df, duplicate_rows = clean_df(df)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = clean_text_columns(df, *TEXT_COLUMNS)
    df = convert_to_numeric(df, *NUMERIC_COLUMNS)
    df = convert_to_date(df, *DATE_COLUMNS)
    df = convert_to_category(df, *CATEGORY_COLUMNS)
    df = convert_to_string(df, *STRING_COLUMNS)
    df["authors_clean"] = df["authors"].apply(clean_authors)
    df["subject_areas"] = df["subject_areas"].str.replace(
        r"\([^)]*\)", "", regex=True)
    output_file = OUTPUT_DIR / "scopus_all_outcomes_clean.csv"
    df.to_csv(output_file, index = False, encoding = "utf-8")

    print("Scopus outcome data cleaning completed.")
    print("=" * 40)
    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Saved          : {output_file.name}")
    print("=" * 40)

    duplicate_keys = duplicate_rows[["project_id", "scopus_id"]]

    inst_file = INPUT_DIR / "scopus_outcomes_institutions_latest.csv"
    if inst_file.exists():
        inst_df = pd.read_csv(inst_file, encoding="utf-8")
        # Remove the same duplicates
        inst_df = inst_df.merge(duplicate_keys,
            on=["project_id", "scopus_id"],
            how="left", indicator=True)
        inst_df = inst_df[inst_df["_merge"] == "left_only"].drop(
            columns="_merge")
        inst_df = convert_to_category(inst_df, "city", "country")
        inst_df = convert_to_string(inst_df, "institution_id", "institution", 
                                    *STRING_COLUMNS)
        inst_output_file = OUTPUT_DIR / "scopus_institutions_clean.csv"
        inst_df.to_csv(inst_output_file, index = False, encoding = "utf-8")

        print("Scopus outcome institution data cleaning completed.")
        print("=" * 40)
        print(f"Rows           : {len(inst_df)}")
        print(f"Columns        : {len(inst_df.columns)}")
        print(f"Saved          : {inst_output_file.name}")
        print("=" * 40)

    ref_file = INPUT_DIR / "scopus_references_latest.csv"
    if ref_file.exists():
        ref_df = pd.read_csv(ref_file, encoding="utf-8")
        # Temporarily rename columns to remove duplicates
        ref_df = ref_df.merge(duplicate_keys,
            left_on=["citing_project_id", "citing_scopus_id"],
            right_on=["project_id", "scopus_id"],
            how="left", indicator=True)
        ref_df = ref_df[ref_df["_merge"] == "left_only"
                        ].drop(
                            columns=["project_id", "scopus_id", "_merge"])
        ref_df = convert_to_category(ref_df, "city", "country")
        ref_df = convert_to_numeric(ref_df, "cited_year")
        ref_df = clean_text_columns(ref_df, "citing_title", "cited_title")
        ref_df = convert_to_string(ref_df, "citing_project_id", "citing_grant_reference", 
                                   "citing_eid", "citing_doi", "cited_doi", "cited_source",
                                   "cited_authors", "reference_text")
        df = df.dropna(axis=1, how="all")
        ref_output_file = OUTPUT_DIR / "scopus_references_clean.csv"
        ref_df.to_csv(ref_output_file, index = False, encoding = "utf-8")

        print("Scopus outcome references data cleaning completed.")
        print("=" * 40)
        print(f"Rows           : {len(ref_df)}")
        print(f"Columns        : {len(ref_df.columns)}")
        print(f"Saved          : {ref_output_file.name}")
        print("=" * 40)


if __name__ == "__main__":
    main()