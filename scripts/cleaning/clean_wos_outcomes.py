"""
Clean processed Web of Science outcome datasets.

The script:
    1. Cleans outcome metadata extracted from the WoS Expanded API.
    2. Standardises data types and text fields.
    3. Removes duplicate project-outcome matches.
    4. Regenerates the deduplicated paper-level table from the cleaned rows.
    5. Produces cleaned outcome, unique-paper and institution datasets.

Exported outputs:
    - wos_all_outcomes_clean.csv    - cleaned project-paper attribution table.
    - wos_outcomes_unique_clean.csv - cleaned paper-level table (one row per
                                      paper, however many grants acknowledge it).
    - wos_institutions_clean.csv    - cleaned author affiliation institutions.
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

INPUT_DIR = ROOT_DIR / "data" / "processed" / "wos"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLEANING CONFIG
# ---------------------------------------------------------------------------

STRING_COLUMNS = [
    "project_id",
    "grant_reference",
    "wos_uid",
    "doi",
    "issn",
    "eissn",
    "source_title",
    "publisher",
    "authors",
    "researcher_ids",
    "orcids",
    "funding_agencies",
    "funding_grant_ids",
    "author_keywords",
    "keywords_plus",
    "wos_categories_traditional",
    "wos_categories_extended",
    "citation_topic_macro",
    "citation_topic_meso",
    "citation_topic_micro",
    "sdg_categories",
    "pub_month",
]

TEXT_COLUMNS = [
    "title",
    "abstract",
    "funding_text",
]

NUMERIC_COLUMNS = [
    "pub_year",
    "early_access_year",
    "times_cited_core",
    "times_cited_all_db",
    "usage_180days",
    "usage_alltime",
    "reference_count",
    "n_addresses",
]

DATE_COLUMNS = [
    "cover_date",
    "sort_date",
]

CATEGORY_COLUMNS = [
    "doctype",
    "open_access_gold",
]

# Nothing is dropped: every collected field is either analytical (citations,
# usage), match evidence (funding text and grant ids) or a quarantined
# taxonomy kept for the publication-side label comparison.
COLS_TO_DROP = []


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
    removed_dupes = pd.DataFrame()
    # Remove duplicate project-outcome matches
    if {"project_id", "wos_uid"}.issubset(df.columns):
        before = len(df)
        # Keep the duplicates that will be removed
        removed_dupes = df[df.duplicated(subset=["project_id", "wos_uid"],
                                         keep="first")]
        # Keep only the first occurrence
        df = df.drop_duplicates(subset=["project_id", "wos_uid"])
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
    processed_file = INPUT_DIR / "wos_outcomes_latest.csv"
    if processed_file.exists():
        input_file = processed_file
    else:
        raise FileNotFoundError(
            "Could not find wos_outcomes_latest.csv")

    df = pd.read_csv(input_file, encoding="utf-8")
    df, duplicate_rows = clean_df(df)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = clean_text_columns(df, *TEXT_COLUMNS)
    df = convert_to_numeric(df, *NUMERIC_COLUMNS)
    df = convert_to_date(df, *DATE_COLUMNS)
    df = convert_to_category(df, *CATEGORY_COLUMNS)
    df = convert_to_string(df, *STRING_COLUMNS)
    df["authors_clean"] = df["authors"].apply(clean_authors)
    output_file = OUTPUT_DIR / "wos_all_outcomes_clean.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print("WoS outcome data cleaning completed.")
    print("=" * 40)
    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Saved          : {output_file.name}")
    print("=" * 40)

    # The paper-level table is regenerated from the cleaned rows rather than
    # cleaned separately, so the two outputs can never disagree. A paper
    # acknowledging several grants appears once, without its attribution
    # columns.
    unique_df = (df.drop_duplicates(subset=["wos_uid"])
                   .drop(columns=["project_id", "grant_reference"],
                         errors="ignore"))
    unique_output_file = OUTPUT_DIR / "wos_outcomes_unique_clean.csv"
    unique_df.to_csv(unique_output_file, index=False, encoding="utf-8")

    print("WoS unique paper table completed.")
    print("=" * 40)
    print(f"Rows           : {len(unique_df)}")
    print(f"Saved          : {unique_output_file.name}")
    print("=" * 40)

    inst_file = INPUT_DIR / "wos_outcomes_institutions_latest.csv"
    if inst_file.exists():
        inst_df = pd.read_csv(inst_file, encoding="utf-8")
        # Remove the same duplicates as the outcome table
        if len(duplicate_rows):
            duplicate_keys = duplicate_rows[["project_id", "wos_uid"]]
            inst_df = inst_df.merge(duplicate_keys,
                                    on=["project_id", "wos_uid"],
                                    how="left", indicator=True)
            inst_df = inst_df[inst_df["_merge"] == "left_only"].drop(
                columns="_merge")
        inst_df = inst_df.replace(TEXT_TO_REPLACE, regex=True)
        inst_df = convert_to_category(inst_df, "city", "country")
        inst_df = convert_to_string(inst_df, "project_id", "grant_reference",
                                    "wos_uid", "institution",
                                    "institution_raw", "full_address")
        inst_output_file = OUTPUT_DIR / "wos_institutions_clean.csv"
        inst_df.to_csv(inst_output_file, index=False, encoding="utf-8")

        print("WoS institution data cleaning completed.")
        print("=" * 40)
        print(f"Rows           : {len(inst_df)}")
        print(f"Columns        : {len(inst_df.columns)}")
        print(f"Saved          : {inst_output_file.name}")
        print("=" * 40)


if __name__ == "__main__":
    main()
