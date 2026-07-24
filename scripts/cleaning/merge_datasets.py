from pathlib import Path
import pandas as pd
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROJECT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
            and float(gtr_funding) != float(oa_funding)
        )
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
    parser.add_argument("--save-differences", action="store_true",
                        help="Save csv indicating where metadata differs.")
    args = parser.parse_args()

    # Load project data
    gtr_file = PROJECT_INPUT_DIR / "gtr_projects_clean.csv"
    openalex_file = PROJECT_INPUT_DIR / "openalex_projects_clean.csv"
    if not gtr_file.exists():
        raise FileNotFoundError(f"GtR dataset not found: {gtr_file}")
    if not openalex_file.exists():
        raise FileNotFoundError(f"OpenAlex dataset not found: {openalex_file}")
    gtr_df = pd.read_csv(gtr_file)
    openalex_df = pd.read_csv(openalex_file)

    # Testing
    print(openalex_df["currency"].unique())
    
    # Compare differences in metadata
    comparison_df = compare_openalex_gtr(gtr_df, openalex_df)

    # Save comparison table if requested
    if args.save_differences:
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



if __name__ == "__main__":
    main()