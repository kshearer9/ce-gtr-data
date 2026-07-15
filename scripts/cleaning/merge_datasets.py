from pathlib import Path
import pandas as pd

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

def merge_projects():
    gtr_df = pd.read_csv(PROJECT_INPUT_DIR / "gtr_projects_clean.csv")
    openalex_df = pd.read_csv(PROJECT_INPUT_DIR / "openalex_projects_clean.csv")

    # Keep only OpenAlex enrichment columns
    openalex_cols = [
        "project_id",
        "primary_topic",
        "primary_topic_score",
        "description_clean"
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    project_df = merge_projects()
    project_output_file = OUTPUT_DIR / "projects.csv"
    project_df.to_csv(project_output_file, index=False, encoding="utf-8")
    
    print("\nMerged GtR and OpenAlex project datasets.")
    print("=" * 40)
    print(f"Rows           : {len(project_df)}")
    print(f"Columns        : {len(project_df.columns)}")
    print(f"Saved          : {project_output_file.name}")
    print("=" * 40)


if __name__ == "__main__":
    main()