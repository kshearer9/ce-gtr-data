from pathlib import Path
import pandas as pd
import argparse
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
# PROJECT MERGE
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
            .fillna(merged_df["funding_type"]))

        # Remove temporary OpenAlex funding type column
        merged_df.drop(columns=["funding_type"], inplace=True, errors="ignore")

    # Replace abstract_text_clean with OpenAlex description if OpenAlex is longer
    if "openalex_description" in merged_df.columns:
        merged_df["abstract_text_clean"] = merged_df.apply(
            lambda row:
                row["openalex_description"]
                if pd.notna(row["openalex_description"])
                and len(str(row["openalex_description"])) 
                > len(str(row.get("abstract_text_clean", "")))
                else row.get("abstract_text_clean"),
            axis=1)

        # Remove temporary OpenAlex description
        merged_df.drop(columns=["openalex_description"], inplace=True, errors="ignore")
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


# ---------------------------------------------------------------------------
# COMPARE PROJECT METADATA
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
            ("end_date_gtr", "end_date_openalex", "end_date")
        ]:
            gtr_date = pd.to_datetime(
                row.get(gtr_col),
                errors="coerce"
            )

            oa_date = pd.to_datetime(
                row.get(oa_col),
                errors="coerce"
            )

            if pd.notna(gtr_date) and pd.notna(oa_date):
                date_difference_days = (
                    oa_date - gtr_date
                ).days
            else:
                date_difference_days = None

            record[f"{label}_difference"] = (
                date_difference_days != 0
                if date_difference_days is not None
                else False
            )

            record[f"{label}_difference_days"] = (
                date_difference_days
            )

            record[f"gtr_{label}"] = gtr_date
            record[f"openalex_{label}"] = oa_date

        comparisons.append(record)
    return pd.DataFrame(comparisons)



# ---------------------------------------------------------------------------
# OUTCOME MERGE
# ---------------------------------------------------------------------------

def merge_outcomes_using_map(gtr_outcomes, openalex_outcomes, scopus_outcomes,
                            wos_outcomes, outcome_map,):
    """
    Build the global outcome dataset using the existing outcome map.

    Each source outcome_id is matched to its corresponding source-specific
    ID in project_outcome_map.csv. The global_outcome_id from the map is
    retained as the unique identifier for each outcome.
    """
    outcomes = outcome_map.copy()
    outcomes.drop(columns=["source", "match_basis", "project_id"], 
                  inplace=True, errors="ignore")
    source_data = [
        ("gtr", gtr_outcomes),
        ("openalex", openalex_outcomes),
        ("scopus", scopus_outcomes),
        ("wos", wos_outcomes),
    ]

    source_id_columns = []
    sources = []
    for source, source_df in source_data:
        source_id_column = f"{source}_outcome_id"
        # Skip sources without a corresponding ID in the map or dataset
        if source_id_column not in outcomes.columns:
            continue
        if "outcome_id" not in source_df.columns:
            continue
        source_df = source_df.copy()
        # Standardise outcome IDs before merging
        outcomes[source_id_column] = (
            outcomes[source_id_column].astype("string").str.strip())
        source_df["outcome_id"] = (
            source_df["outcome_id"].astype("string").str.strip())
        # Keep only unique outcomes
        source_df = source_df.drop_duplicates(subset=["outcome_id"], keep="first")
        source_df = source_df.rename(columns={col: f"{source}_{col}"
                                              for col in source_df.columns
                                              if col != "outcome_id"})
        # Rename the source ID to match the map
        source_df = source_df.rename(columns={"outcome_id": source_id_column})
        source_columns = [source_id_column]
        for column in source_df.columns:
            if column != source_id_column:
                source_columns.append(column)
        source_df = source_df[source_columns]
        # Add source metadata to the global outcome dataset
        outcomes = outcomes.merge(
            source_df,
            on=source_id_column,
            how="left")
        source_id_columns.append(source_id_column)
        sources.append(source)

    # Remove unnecessary columns
    outcomes = outcomes.drop(columns=source_id_columns, errors = "ignore")
    cols_to_remove = [
        "project_id",
        "project_title",
        "project_acronym",
        "project_start_date",
        "project_end_date",
        "grant_category",
        "funding_type",
        "grant_reference",
        "project_openalex_url",
        "source_id",
        "publisher",
        "n_addresses",
        "funding_agencies",
        "funding_grant_ids"
    ]
    source_columns_to_remove = [f"{source}_{column}"
                                for source in sources
                                for column in cols_to_remove]
    outcomes.drop(columns=source_columns_to_remove, inplace=True, 
                  errors="ignore")
    # Replace original columns with their cleaned versions.
    clean_columns = [col for col in outcomes.columns
                     if col.endswith("_clean")
                     and col[:-6] in outcomes.columns]
    for clean_col in clean_columns:
        original_col = clean_col[:-6]
        outcomes[original_col] = outcomes[clean_col]
    outcomes.drop(columns=clean_columns, inplace=True, errors="ignore")



    # Remove source prefixes where no other source has the same field
    for column in list(outcomes.columns):
        for source in sources:
            prefix = f"{source}_"
            if column.startswith(prefix):
                base_column = column[len(prefix):]
                matching_columns = [
                    f"{other_source}_{base_column}"
                    for other_source in sources
                    if other_source != source
                    and f"{other_source}_{base_column}" in outcomes.columns]
                if not matching_columns:
                    outcomes = outcomes.rename(columns={column: base_column})
                break
    return outcomes


# ---------------------------------------------------------------------------
# OUTCOME COLUMN MERGING
# ---------------------------------------------------------------------------

def merge_preferred_column(df, output_column, source_columns,):
    """
    Use the first available non-missing value in source priority order.
    """
    df[output_column] = pd.NA
    for column in source_columns:
        if column not in df.columns:
            continue
        df[output_column] = (df[output_column].fillna(df[column]))
    return df



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
    gtr_df = pd.read_csv(gtr_file, encoding = "utf-8")
    openalex_df = pd.read_csv(openalex_file, encoding = "utf-8")
    
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
    
    print()
    print("=" * 70)
    print("PROJECT MERGE SUMMARY")
    print("=" * 70)
    print(f"Rows           : {len(project_df)}")
    print(f"Columns        : {len(project_df.columns)}")
    print(f"Saved          : {project_output_file.name}")
    print("\nOpenAlex-GtR Project Comparison Summary:")
    print("-" * 70)    
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
    print("-" * 70)

    print("\nStart date difference distribution:")
    print("-" * 70)
    print(
        comparison_df["start_date_difference_days"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nEnd date difference distribution:")
    print("-" * 70)
    print(
        comparison_df["end_date_difference_days"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # ------------------------------------------------
    # OUTCOMES
    # ------------------------------------------------

    gtr_outcome_file = OUTCOME_INPUT_DIR / "gtr_all_outcomes_clean.csv"
    openalex_outcome_file = OUTCOME_INPUT_DIR / "openalex_all_outcomes_clean.csv"
    scopus_outcome_file = OUTCOME_INPUT_DIR / "scopus_all_outcomes_clean.csv"
    wos_outcome_file = OUTCOME_INPUT_DIR / "wos_all_outcomes_clean.csv"

    # Load
    gtr_outcome_df = pd.read_csv(gtr_outcome_file, encoding="utf-8",
                                 dtype={"outcome_id": "string"})
    openalex_outcome_df = pd.read_csv(openalex_outcome_file, encoding="utf-8",
                                      dtype={"outcome_id": "string"})
    scopus_outcome_df = pd.read_csv(scopus_outcome_file, encoding="utf-8",
                                    dtype={"outcome_id": "string"})
    wos_outcome_df = pd.read_csv(wos_outcome_file, encoding="utf-8",
                                 dtype={"outcome_id": "string"})
    
    # Merge outcomes
    outcome_map_file = OUTPUT_DIR / "project_outcome_map.csv"
    if not outcome_map_file.exists():
        raise FileNotFoundError(
            f"Project-outcome map not found: {outcome_map_file}")
    outcome_map = pd.read_csv(outcome_map_file, encoding="utf-8",
                               dtype={"gtr_outcome_id": "string",
                                      "openalex_outcome_id": "string",
                                      "scopus_outcome_id": "string",
                                      "wos_outcome_id": "string"})
    outcome_df = merge_outcomes_using_map(
        gtr_outcomes=gtr_outcome_df,
        openalex_outcomes=openalex_outcome_df,
        scopus_outcomes=scopus_outcome_df,
        wos_outcomes=wos_outcome_df,
        outcome_map=outcome_map,
    )

    # ------------------------------------------------------------------
    # PRINT REPORT
    # ------------------------------------------------------------------

    outcome_output_file = OUTPUT_DIR / "outcomes.csv"
    
    """print_outcome_report(
        outcome_df,
        outcome_output_file
    )"""

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    outcome_df.to_csv(
        outcome_output_file,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved outcomes to: "
        f"{outcome_output_file}"
    )


if __name__ == "__main__":
    main()