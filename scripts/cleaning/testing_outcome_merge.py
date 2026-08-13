from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import re
from utils.merge.type_mappings import GTR_TYPE_MAP, OPENALEX_TYPE_MAP
from utils.merge.col_mappings import OPENALEX_COL_MAP
from utils.cleaning import create_outcome_map

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTCOME_INPUT_DIR = ROOT_DIR / "team-code" / "data" / "cleaned" / "outcomes"

OUTPUT_DIR = ROOT_DIR / "testing" / "merge"

for d in (OUTCOME_INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
    
# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

def normalise_title(title):
    """
    Normalise titles for matching.
    """
    if pd.isna(title):
        return ""

    title = str(title).lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)

    return title.strip()

def normalise_identifier(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .replace("-", "")
        .replace(" ", "")
    )

def normalise_outcome_type(df, source):
    """
    Map source-specific outcome types to canonical type and subtype.
    """
    if source == "gtr":
        type_map = GTR_TYPE_MAP
    elif source == "openalex":
        type_map = OPENALEX_TYPE_MAP
    else:
        raise ValueError("source must be 'gtr' or 'openalex'")
    mapped = df["type"].map(type_map)
    df["type_normalised"] = (
        mapped
        .apply(lambda x: x["type"] if isinstance(x, dict) else None)
        .fillna("other"))
    df["subtype_normalised"] = (
        mapped
        .apply(lambda x: x["subtype"] if isinstance(x, dict) else None)
        .fillna("other"))
    return df

def expand_grant_references(df):
    """
    Expand semicolon-separated grant references into rows
    for matching purposes only.
    """

    df = df.copy()

    df["grant_reference_list"] = (
        df["grant_reference"]
        .fillna("")
        .str.split(";")
    )

    df = df.explode(
        "grant_reference_list"
    )

    df["grant_reference_list"] = (
        df["grant_reference_list"]
        .apply(normalise_identifier)
    )

    return df

def create_match_key(df):
    """
    Create title + grant reference matching key.
    """

    df["title_key"] = (
        df["title_clean"]
        .apply(normalise_title)
    )

    df["grant_key"] = (
        df["grant_reference_list"]
    )

    df["match_key"] = (
        df["title_key"]
        + "|"
        + df["grant_key"]
    )

    return df

def append_new_outcomes(
    master_df,
    external_df,
    column_mapping,
    source_name,
    output_dir
):

    external = external_df.rename(
        columns=column_mapping
    )

    master = expand_grant_references(master_df)
    external = expand_grant_references(external)

    master = create_match_key(master)
    external = create_match_key(external)

    existing_keys = set(
        master["match_key"]
    )

    new_records = external[
        ~external["match_key"].isin(existing_keys)
    ]

    matched_records = external[
        external["match_key"].isin(existing_keys)
    ]

    new_records.to_csv(
        output_dir /
        f"{source_name}_new_outcomes.csv",
        index=False
    )

    combined = pd.concat(
        [
            master_df,
            new_records.drop(
                columns=[
                    "grant_reference_list",
                    "title_key",
                    "grant_key",
                    "match_key"
                ],
                errors="ignore"
            )
        ],
        ignore_index=True
    )

    return combined, matched_records, new_records


# ---------------------------------------------------------------------------
# COMPARE METADATA
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save csv comparing metadata.")
    args = parser.parse_args()

    maps = []

    # -------------------------
    # GtR
    # -------------------------

    gtr = pd.read_csv(
        OUTCOME_INPUT_DIR /
        "gtr_all_outcomes_clean.csv"
    )

    gtr_map = create_outcome_map(
        gtr,
        source="gtr",
        id_column="outcome_id"
    )

    maps.append(gtr_map)


    # -------------------------
    # OpenAlex
    # -------------------------

    openalex = pd.read_csv(
        OUTCOME_INPUT_DIR /
        "openalex_all_outcomes_clean.csv"
    )

    openalex_map = create_outcome_map(
        openalex,
        source="openalex",
        id_column="outcome_id"
    )

    maps.append(openalex_map)


    # Combine
    outcome_map = pd.concat(
        maps,
        ignore_index=True
    )
    outcome_map = (
        outcome_map
        .groupby(
            [
                "grant_reference",
                "outcome_title"
            ],
            as_index=False
        )
        .agg({
            "gtr_id": "first",
            "openalex_id": "first"
        })
    )


    # Save
    outcome_map.to_csv(
        OUTPUT_DIR /
        "grant_outcome_map.csv",
        index=False
    )


    print(
        f"Saved {len(outcome_map)} mappings"
    )

    print("GtR map:", len(gtr_map))
    print("OpenAlex map:", len(openalex_map))

    outcome_map = pd.concat(
        maps,
        ignore_index=True
    )

    print("Before grouping:", len(outcome_map))

    duplicates = (
        outcome_map
        .groupby(
            [
                "grant_reference",
                "outcome_title"
            ]
        )
        .size()
        .reset_index(name="count")
    )

    print(
        duplicates[
            duplicates["count"] > 1
        ]
        .head(20)
    )

    def combine_ids(series):
        return "; ".join(
            series
            .dropna()
            .astype(str)
            .unique()
        )


    outcome_map = (
        outcome_map
        .groupby(
            [
                "grant_reference",
                "outcome_title"
            ],
            as_index=False
        )
        .agg({
            "gtr_id": combine_ids,
            "openalex_id": combine_ids
        })
    )

    print("After grouping:", len(outcome_map))

    # ------------------------------------------------
    # OUTCOMES
    # ------------------------------------------------
    # Load project data
    gtr_outcome_file = OUTCOME_INPUT_DIR / "gtr_all_outcomes_clean.csv"
    openalex_outcome_file = OUTCOME_INPUT_DIR / "openalex_all_outcomes_clean.csv"
    if not gtr_outcome_file.exists():
        raise FileNotFoundError(f"GtR dataset not found: {gtr_outcome_file}")
    if not openalex_outcome_file.exists():
        raise FileNotFoundError(f"OpenAlex dataset not found: {openalex_outcome_file}")
    gtr_outcome_df = pd.read_csv(gtr_outcome_file)
    openalex_outcome_df = pd.read_csv(openalex_outcome_file)
    merged_df, matched, new = append_new_outcomes(
        master_df=gtr_outcome_df,
        external_df=openalex_outcome_df,
        column_mapping=OPENALEX_COL_MAP,
        source_name="openalex",
        output_dir=OUTPUT_DIR
    )

    # Save final combined file
    output_file = (
        OUTPUT_DIR /
        "outcomes_merged.csv"
    )

    merged_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print("=" * 40)
    print(
        f"Original GtR outcomes : {len(gtr_outcome_df)}"
    )
    print(
        f"OpenAlex duplicates   : {len(matched)}"
    )
    print(
        f"New OpenAlex outcomes : {len(new)}"
    )
    print(
        f"Final combined file  : {len(merged_df)}"
    )
    print(
        f"Saved                : {output_file.name}"
    )
    print("=" * 40)

if __name__ == "__main__":
    main()