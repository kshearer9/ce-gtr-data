from pathlib import Path
import pandas as pd
import argparse
import numpy as np
from utils.merge.type_mappings import GTR_TYPE_MAP, OPENALEX_TYPE_MAP
from utils.merge.col_mappings import OPENALEX_COL_MAP

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROJECT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTCOME_INPUT_DIR = PROJECT_INPUT_DIR / "outcomes"

OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "merged"

for d in (PROJECT_INPUT_DIR, OUTCOME_INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

def merge_projects(gtr_df, openalex_df):
    # Keep only OpenAlex enrichment columns
    openalex_cols = [
        "project_id",
        "primary_topic",
        "primary_topic_score",
        "description_clean",
        "subfield",
        "field",
        "domain"]
    openalex_df = openalex_df[
        [col for col in openalex_cols if col in openalex_df.columns]]

    # Rename OpenAlex description before merging to avoid collision
    if "description_clean" in openalex_df.columns:
        openalex_df = openalex_df.rename(
            columns={"description_clean": "openalex_description"})

    # Left join - keep every GTR project
    merged_df = gtr_df.merge(
        openalex_df,
        on="project_id",
        how="left")

    # If fund type is missing in gtr, replace with openalex value
    if "funding_type" in merged_df.columns and "grant_category" in merged_df.columns:
        merged_df["grant_category"] = (
            merged_df["grant_category"]
            .fillna(merged_df["funding_type"])
        )

        # Remove temporary OpenAlex funding type column
        merged_df.drop(
            columns=["funding_type"],
            inplace=True,
            errors="ignore"
        )

    # Replace abstract_text_clean with OpenAlex description if OpenAlex is longer
    if "openalex_description" in merged_df.columns:
        merged_df["abstract_text_clean"] = merged_df.apply(
            lambda row:
                row["openalex_description"]
                if pd.notna(row["openalex_description"])
                and len(str(row["openalex_description"])) 
                > len(str(row.get("abstract_text_clean", "")))
                else row.get("abstract_text_clean"),
            axis=1
        )

        # Remove temporary OpenAlex description
        merged_df.drop(columns=["openalex_description"],
                       inplace=True, errors="ignore")
        # Remove old unclean abstract field
        merged_df.drop(columns=["abstract_text"], inplace=True, errors="ignore")
    # Remove original columns if cleaned version exists
    cleaned_cols = [col for col in merged_df.columns 
                    if col.endswith("_clean")]
    originals_to_remove = [col.replace("_clean", "")
                           for col in cleaned_cols
                           if col.replace("_clean", "") in merged_df.columns]
    merged_df.drop(columns=originals_to_remove, inplace=True, errors="ignore")
    return merged_df

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

def append_external_records(
    gtr_df,
    external_df,
    column_mapping,
    source_name=None
):
    """
    Append records from an external dataset into GtR format.

    Parameters:
    gtr_df: existing GtR dataframe
    external_df: dataframe to append
    column_mapping: dictionary mapping external columns to GtR columns
    source_name: optional name of source dataset

    Returns:
    Combined dataframe
    """

    external = external_df.copy()

    # Rename external columns to GtR names
    external = external.rename(columns=column_mapping)

    # Keep only columns that exist in GtR
    common_cols = [
        col for col in external.columns
        if col in gtr_df.columns
    ]

    external = external[common_cols]

    # Add missing GtR columns to external dataframe
    for col in gtr_df.columns:
        if col not in external.columns:
            external[col] = pd.NA

    # Ensure same column order
    external = external[gtr_df.columns]

    # Optional tracking
    if source_name:
        external["source"] = source_name
        if "source" not in gtr_df.columns:
            gtr_df["source"] = "gtr"

    combined = pd.concat(
        [gtr_df, external],
        ignore_index=True
    )

    return combined

def merge_openalex_outcomes(gtr_outcomes, openalex_outcomes):

    oa_cols = [
        "title_clean",
        "doi",
        "type",
        "cited_by",
        "fwci",
        "topics",
        "domain",
        "field",
        "subfield"
    ]

    openalex = openalex_outcomes[
        [c for c in oa_cols if c in openalex_outcomes.columns]
    ].copy()

    gtr = gtr_outcomes.copy()

    # Normalise types
    gtr = normalise_outcome_type(gtr, "gtr")
    openalex = normalise_outcome_type(openalex, "openalex")

    print("\nOpenAlex outcome matching")
    print("-------------------------")
    print(f"OpenAlex records: {len(openalex)}")
    print(f"GtR records: {len(gtr)}")

    # Remove duplicate OpenAlex records with identical matching keys
    openalex_duplicates = (
        openalex.groupby(
            ["title_clean", "doi", "type_normalised"]
        )
        .size()
        .reset_index(name="count")
    )

    openalex_duplicates = openalex_duplicates[
        openalex_duplicates["count"] > 1
    ]

    print(
        "Duplicate OpenAlex title/DOI/type combinations removed:",
        len(openalex_duplicates)
    )

    openalex = openalex.drop_duplicates(
        subset=[
            "title_clean",
            "doi",
            "type_normalised"
        ],
        keep="first"
    )

    # Match only on title + doi + type
    matched = openalex.merge(
        gtr,
        on=[
            "title_clean",
            "doi",
            "type_normalised"
        ],
        how="inner",
        suffixes=("_openalex", "_gtr")
    )

    unmatched = openalex.merge(
        gtr[
            [
                "title_clean",
                "doi",
                "type_normalised"
            ]
        ],
        on=[
            "title_clean",
            "doi",
            "type_normalised"
        ],
        how="left",
        indicator=True
    )

    unmatched = unmatched[
        unmatched["_merge"] == "left_only"
    ].drop(columns="_merge")

    print(f"Matched records: {len(matched)}")
    print(f"Unmatched OpenAlex records: {len(unmatched)}")

    # Add enrichment to ALL matching GtR records
    enrichment_cols = [
        "title_clean",
        "doi",
        "type_normalised",
        "cited_by",
        "fwci",
        "topics",
        "domain",
        "field",
        "subfield"
    ]

    gtr = gtr.merge(
        matched[enrichment_cols].drop_duplicates(
            subset=[
                "title_clean",
                "doi",
                "type_normalised"
            ]
        ),
        on=[
            "title_clean",
            "doi",
            "type_normalised"
        ],
        how="left"
    )

    print(
        "OpenAlex enrichment added:",
        gtr["topics"].notna().sum()
    )

    # Get full OpenAlex records that did not match
    unmatched_keys = unmatched[
        [
            "title_clean",
            "doi",
            "type_normalised"
        ]
    ]

    # Create full OpenAlex dataframe with normalised type for matching
    openalex_full = openalex_outcomes.copy()
    openalex_full = normalise_outcome_type(openalex_full, "openalex")

    openalex_unmatched_full = openalex_full.merge(
        unmatched_keys,
        on=[
            "title_clean",
            "doi",
            "type_normalised"
        ],
        how="inner"
    )

    print(
        "Full OpenAlex records appended:",
        len(openalex_unmatched_full)
    )

    outcomes = append_external_records(
        gtr,
        openalex_unmatched_full,
        OPENALEX_COL_MAP,
        source_name="openalex"
    )

    return outcomes

        




# ---------------------------------------------------------------------------
# COMPARE METADATA
# ---------------------------------------------------------------------------

def compare_openalex_gtr(gtr_df, openalex_df):
    # Harmonise OpenAlex funding types with GtR grant categories
    if "funding_type" in openalex_df.columns:
        openalex_df["funding_type"] = openalex_df["funding_type"].replace({
            "research": "research grant",
            "voucher": "vouchers",
            "training": "training grant"})

    # Merge datasets
    merged = gtr_df.merge(openalex_df, on="project_id",
                          how="inner", suffixes=("_gtr", "_openalex"))

    comparisons = []
    for _, row in merged.iterrows():
        record = {"project_id": row["project_id"]}

        # Compare descriptions
        gtr_desc = row.get("abstract_text_clean")
        oa_desc = row.get("description_clean")
        record["gtr_description_length"] = (
            len(str(gtr_desc)) if pd.notna(gtr_desc) else 0)
        record["openalex_description_length"] = (
            len(str(oa_desc)) if pd.notna(oa_desc) else 0)
        record["openalex_description_longer"] = (
            pd.notna(oa_desc)
            and len(str(oa_desc)) > len(str(gtr_desc)))
        record["description_difference"] = (
            str(gtr_desc) != str(oa_desc))

        # Compare funding amount
        gtr_funding = row.get("value_gbp_gtr")
        oa_funding = row.get("value_gbp_openalex")
        record["funding_difference"] = (
            pd.notna(gtr_funding)
            and pd.notna(oa_funding)
            and float(gtr_funding) != float(oa_funding))
        record["gtr_funding"] = gtr_funding
        record["openalex_funding"] = oa_funding

        # Compare funding type/category
        gtr_type = row.get("grant_category")
        oa_type = row.get("funding_type")
        record["funding_type_difference"] = (
            pd.notna(gtr_type)
            and pd.notna(oa_type)
            and str(gtr_type).lower() != str(oa_type).lower())
        record["gtr_grant_category"] = gtr_type
        record["openalex_funding_type"] = oa_type

        # Compare dates
        for gtr_col, oa_col, label in [
            ("start_date_gtr", "start_date_openalex", "start_date"),
            ("end_date_gtr", "end_date_openalex", "end_date")]:
            gtr_date = pd.to_datetime(row.get(gtr_col), errors="coerce")
            oa_date = pd.to_datetime(row.get(oa_col), errors="coerce")
            record[f"{label}_difference"] = (pd.notna(gtr_date)
                and pd.notna(oa_date) and gtr_date != oa_date)
            record[f"gtr_{label}"] = gtr_date
            record[f"openalex_{label}"] = oa_date

        comparisons.append(record)
    return pd.DataFrame(comparisons)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save csv comparing metadata.")
    args = parser.parse_args()

    # ------------------------------------------------
    # PROJECTS
    # ------------------------------------------------

    # Load project data
    gtr_file = PROJECT_INPUT_DIR / "gtr_projects_clean.csv"
    openalex_file = PROJECT_INPUT_DIR / "openalex_projects_clean.csv"
    if not gtr_file.exists():
        raise FileNotFoundError(f"GtR dataset not found: {gtr_file}")
    if not openalex_file.exists():
        raise FileNotFoundError(f"OpenAlex dataset not found: {openalex_file}")
    gtr_df = pd.read_csv(gtr_file)
    openalex_df = pd.read_csv(openalex_file)
    
    # Compare differences in metadata
    comparison_df = compare_openalex_gtr(gtr_df, openalex_df)

    # Save comparison table if requested
    if args.save_comparison:
        comparison_file = OUTPUT_DIR / "project_metadata_comparison.csv"
        comparison_df.to_csv(
            comparison_file,
            index=False, encoding="utf-8")
        print(f"Saved comparison table as {comparison_file.name}")
    
    project_df = merge_projects(gtr_df, openalex_df)
    project_output_file = OUTPUT_DIR / "projects.csv"
    project_df.to_csv(project_output_file, index=False, encoding="utf-8")
    
    print("\nMerged GtR and OpenAlex project datasets.")
    print("=" * 40)
    print(f"Rows           : {len(project_df)}")
    print(f"Columns        : {len(project_df.columns)}")
    print(f"Saved          : {project_output_file.name}")
    print("=" * 40)
    print("\nOpenAlex-GtR Comparison Summary:")
    print("=" * 40)
    summary_labels = {
        "openalex_description_longer": "OpenAlex Longer Description",
        "description_difference": "Description Difference",
        "funding_difference": "Funding Difference",
        "funding_type_difference": "Funding Type Difference",
        "start_date_difference": "Start Date Difference",
        "end_date_difference": "End Date Difference"
    }
    for col, label in summary_labels.items():
        if col in comparison_df.columns:
            print(f"{label:<30}: {comparison_df[col].sum()}")
    print("=" * 40)

    # ------------------------------------------------
    # OUTCOMES
    # ------------------------------------------------
    # Load project data
    gtr_outcome_file = OUTCOME_INPUT_DIR / "gtr_all_outcomes_clean.csv"
    openalex_outcome_file = OUTCOME_INPUT_DIR / "openalex_all_outcomes_clean.csv"
    if not gtr_file.exists():
        raise FileNotFoundError(f"GtR dataset not found: {gtr_outcome_file}")
    if not openalex_file.exists():
        raise FileNotFoundError(f"OpenAlex dataset not found: {openalex_outcome_file}")
    gtr_outcome_df = pd.read_csv(gtr_outcome_file)
    outcome_df = gtr_outcome_df
    openalex_outcome_df = pd.read_csv(openalex_outcome_file)
    outcome_df = (
        merge_openalex_outcomes(
            outcome_df,
            openalex_outcome_df
        )
    )
    print(outcome_df["type"].notna().sum())
    print(outcome_df["type_normalised"].notna().sum())
    same_title_doi = (
        outcome_df
        .groupby(["title_clean", "doi"])
        .size()
        .reset_index(name="count")
    )

    duplicates = same_title_doi[same_title_doi["count"] > 1]

    print(f"Duplicate title + DOI combinations: {len(duplicates)}")
    print(f"Records involved: {duplicates['count'].sum()}")
    outcome_output_file = OUTPUT_DIR / "outcomes.csv"
    outcome_df.to_csv(outcome_output_file, index=False, encoding="utf-8")




if __name__ == "__main__":
    main()