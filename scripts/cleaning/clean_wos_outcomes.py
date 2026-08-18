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
import re
from utils.cleaning import (normalise_name, convert_to_string,
                            clean_text_columns,
                            convert_to_category, convert_to_numeric)
from utils.constants import TEXT_TO_REPLACE
from utils.col_types import OUTCOME_COLUMN_TYPES, read_csv


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
    "outcome_id",
    "doi",
    "issn",
    "eissn",
    "source_title",
    "publisher",
    "author",
    "researcher_ids",
    "orcids",
    "funding_agencies",
    "funding_grant_ids",
    "author_keywords",
    "keywords",
    "categories_traditional",
    "categories_extended",
    "citation_topic_macro",
    "citation_topic_meso",
    "citation_topic_micro",
    "sdg_categories",
    "year",
    "early_access_year"
]

TEXT_COLUMNS = [
    "title",
    "abstract",
    "funding_text"
]

NUMERIC_COLUMNS = [
    # collect_wos.py emitted `times_cited_all_db` until 17 August and
    # `cited_by` after it. Both are listed so files collected either side of
    # that change clean correctly.
    "times_cited_all_db",
    "cited_by",
    "reference_count",
    "n_addresses"
]

DATE_COLUMNS = [
    "cover_date",
    "sort_date"
]

CATEGORY_COLUMNS = [
    "type",
    "open_access_gold"
]

# Dropping project-related fields
COLS_TO_DROP = ["funding_agencies",
                "funding_grant_ids",
                "funding_text",
                "n_addresses",
                "early_access_year",
                "open_access_gold",
                "times_cited_core"]


# ---------------------------------------------------------------------------
# CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

def clean_author(author):
    """
    Normalise semicolon-separated author names.
    """
    if pd.isna(author):
        return pd.NA
    cleaned = []
    for name in str(author).split(";"):
        name = name.strip()
        if not name:
            continue
        normalised = normalise_name(name)
        if normalised:
            cleaned.append(normalised)
    return "; ".join(cleaned) if cleaned else pd.NA

def clean_wos_outcome_id(value): 
    """ Remove the WOS: prefix from a WoS UID. 
    Example: WOS:001794901400001 -> 001794901400001 """ 
    if pd.isna(value): 
        return pd.NA 
    value = str(value).strip() 
    if value.upper().startswith("WOS:"): 
        value = value[4:] 
    return value

def make_doi_url(doi):
    """
    Create a DOI URL from a DOI value.
    """
    if pd.isna(doi):
        return pd.NA
    doi = str(doi).strip()
    if not doi:
        return pd.NA
    # Remove an existing DOI URL if present.
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    # Remove a leading "doi:" if present.
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return f"https://doi.org/{doi}"

def clean_issn(value):
    """
    Clean ISSN/EISSN values.
    Ensure 8 characters, hyphen is present 
    and returns multiples separated by ";"
    """
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().upper()
    if not value:
        return pd.NA
    # Find possible ISSNs, with or without a hyphen.
    matches = re.findall(r'(?<!\d)\d{4}-?\d{3}[0-9X](?!\d)',
                         value)
    if not matches:
        return pd.NA
    cleaned = []
    for issn in matches:
        # Remove existing hyphen
        issn = issn.replace("-", "")
        # Must be exactly 8 characters
        if len(issn) != 8:
            continue
        # Reinsert the hyphen
        issn = f"{issn[:4]}-{issn[4:]}"
        cleaned.append(issn)
    # Remove duplicates while preserving order
    cleaned = list(dict.fromkeys(cleaned))
    return "; ".join(cleaned) if cleaned else pd.NA


def clean_df(df):
    removed_dupes = pd.DataFrame()
    # Rename Wos UID to outcome_id
    if "wos_uid" in df.columns: 
        df = df.rename(columns={"wos_uid": "outcome_id"})
    # Remove WOS: prefix
    if "outcome_id" in df.columns: 
        df["outcome_id"] = df["outcome_id"].apply(clean_wos_outcome_id)
    # Remove duplicate project-outcome matches
    if {"project_id", "outcome_id"}.issubset(df.columns):
        before = len(df)
        # Keep the duplicates that will be removed
        removed_dupes = df[df.duplicated(subset=["project_id", "outcome_id"],
                                         keep="first")]
        # Keep only the first occurrence
        df = df.drop_duplicates(subset=["project_id", "outcome_id"])
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} duplicate outcomes")
    # Replace placeholder text with NaN
    df = df.replace(TEXT_TO_REPLACE, regex=True)
    # Remove leading/trailing whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df, removed_dupes

def drop_empty_columns(df):
    """Drop columns containing no non-empty values."""
    empty_columns = []
    for column in df.columns:
        values = df[column]
        # Missing values
        if values.isna().all():
            empty_columns.append(column)
            continue
        # Also treat empty/whitespace-only strings as empty
        non_empty = values.notna() & values.astype(str).str.strip().ne("")
        if not non_empty.any():
            empty_columns.append(column)
    if empty_columns:
        print()
        print("Dropped empty columns:")
        for column in empty_columns:
            print(f"  - {column}")
        df = df.drop(columns=empty_columns)
    return df


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

    df = read_csv(input_file, OUTCOME_COLUMN_TYPES)
    df, duplicate_rows = clean_df(df)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = clean_text_columns(df, *TEXT_COLUMNS)
    # Normalise the citation column name. The collector was renamed on
    # 17 August; older collected files still carry the old name, and the
    # merge keys on `cited_by`.
    if "times_cited_all_db" in df.columns and "cited_by" not in df.columns:
        df = df.rename(columns={"times_cited_all_db": "cited_by"})
    df = convert_to_numeric(df, *NUMERIC_COLUMNS)
    # WoS dates are ISO strings (2025-10-10), unlike GtR's millisecond
    # timestamps, so the shared convert_to_date helper (unit="ms") coerces
    # them all to NaT. Parse them directly instead.
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = (pd.to_datetime(df[col], format="mixed", 
                                     errors="coerce"))
    # Use sort-date as a fallback where cover-date is missing
    if "cover_date" in df.columns and "sort_date" in df.columns:
        df["cover_date"] = df["cover_date"].fillna(df["sort_date"])
    # Rename the final publication date field
    df = df.rename(columns={"cover_date": "publication_date"})
    # Create a general publication year field from publication_date
    if "publication_date" in df.columns:
        df["year"] = df["publication_date"].dt.year
    # Remove the other date fields no longer needed
    df = df.drop(columns=["sort_date", "pub_year", "pub_month"], errors="ignore")
    df = convert_to_category(df, *CATEGORY_COLUMNS)
    if "issn" in df.columns:
        df["issn"] = df["issn"].apply(clean_issn)
    if "eissn" in df.columns:
        df["eissn"] = df["eissn"].apply(clean_issn)
    df = convert_to_string(df, *STRING_COLUMNS)
    # Create a DOI URL from the cleaned DOI.
    if "doi" in df.columns:
        df["url"] = df["doi"].apply(make_doi_url)
    df["author_clean"] = df["author"].apply(clean_author)
    df = drop_empty_columns(df)
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
    unique_df = (df.drop_duplicates(subset=["outcome_id"])
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
        # Clean WoS UID again
        if "wos_uid" in inst_df.columns: 
            inst_df = inst_df.rename(columns={"wos_uid": "outcome_id"}) 
        if "outcome_id" in inst_df.columns:
            inst_df["outcome_id"] = (inst_df["outcome_id"] 
                                     .apply(clean_wos_outcome_id))
        # Remove the same duplicates as the outcome table
        if len(duplicate_rows):
            duplicate_keys = duplicate_rows[["project_id", "outcome_id"]]
            inst_df = inst_df.merge(duplicate_keys,
                                    on=["project_id", "outcome_id"],
                                    how="left", indicator=True)
            inst_df = inst_df[inst_df["_merge"] == "left_only"].drop(
                columns="_merge")
        inst_df = inst_df.replace(TEXT_TO_REPLACE, regex=True)
        inst_df = convert_to_category(inst_df, "city", "country")
        inst_df = convert_to_string(inst_df, "project_id", "grant_reference",
                                    "outcome_id", "institution",
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
