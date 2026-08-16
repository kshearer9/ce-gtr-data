"""
Merge cleaned project and outcome data from all source datasets.

The script:
    1. Merges GtR and OpenAlex project records.
    2. Loads and merges outcome records from GtR, OpenAlex, Scopus and WoS.
    3. Uses the existing project-outcome mapping to align outcome records.
    4. Validates source URLs before selecting the final URL by default.
    5. Saves the final merged projects and outcomes datasets.

URL validation behaviour:
    By default, an existing URL validation cache is reused. If no cache
    exists, URL validation is run automatically.
    --no-url-validation
        Skip URL validation and select the first available URL according
        to source priority.
    --refresh-url-validation
        Run the complete URL validation process again, replacing the
        existing validation cache.

Examples:
    python3 -m scripts.cleaning.merge
    python3 -m scripts.cleaning.merge --no-url-validation
    python3 -m scripts.cleaning.merge --refresh-url-validation
"""

import pandas as pd
import argparse
from utils.col_types import PROJECT_COLUMN_TYPES, OUTCOME_COLUMN_TYPES, read_csv
from utils.merge_config import (
    OUTPUT_DIR,
    PROJECT_DISAGREEMENT_DIR,
    PROJECT_INPUT_DIR,
    OUTCOME_INPUT_DIR,
    ensure_directories,
)
from scripts.cleaning.merge_helpers.merge_projects import (
    merge_projects,
    compare_projects
)
from scripts.cleaning.merge_helpers.merge_outcomes import merge_outcomes
from scripts.cleaning.merge_helpers.url_utils import(
    ensure_url_validation_exists,
    load_url_validation
)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ensure_directories()
    parser = argparse.ArgumentParser(description="Merge project and outcome datasets.")
    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument("--no-url-validation", action="store_true",
                        help=("Skip URL validation and use the first available "
                              "URL according to source priority."))
    url_group.add_argument("--refresh-url-validation", action="store_true",
                        help=("Run URL validation again even if a validation "
                              "cache already exists."))
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # PROJECTS
    # ------------------------------------------------------------------

    gtr_file = PROJECT_INPUT_DIR / "gtr_projects_clean.csv"
    openalex_file = PROJECT_INPUT_DIR / "openalex_projects_clean.csv"
    if not gtr_file.exists():
        raise FileNotFoundError(f"GtR dataset not found: {gtr_file}")
    if not openalex_file.exists():
        raise FileNotFoundError(
            f"OpenAlex dataset not found: {openalex_file}")

    gtr_df = read_csv(gtr_file, PROJECT_COLUMN_TYPES)
    openalex_df = read_csv(openalex_file, PROJECT_COLUMN_TYPES)

    print()
    print("=" * 70)
    print("PROJECT MERGE SUMMARY")
    print("=" * 70)
    project_df = merge_projects(gtr_df, openalex_df)
    project_disagreement = compare_projects(gtr_df, openalex_df)
    # Save project disagreements
    for column, disagreement_df in project_disagreement.items():
        if disagreement_df.empty:
            print(f"{'Disagreements Saved':<35}: None for {column}")
            continue
        disagreement_file = (
            PROJECT_DISAGREEMENT_DIR / f"{column}_disagreements.csv")
        disagreement_df.to_csv(disagreement_file, index=False,
                               encoding="utf-8")
    print(f"\nSaved disagreements to {PROJECT_DISAGREEMENT_DIR}")
    project_output_file = OUTPUT_DIR / "projects.csv"

    # Save merged projects
    project_df.to_csv(project_output_file, index=False, encoding="utf-8")
    print(f"\nSaved projects to: {project_output_file}")

    # ------------------------------------------------------------------
    # OUTCOMES
    # ------------------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("OUTCOME MERGE SUMMARY")
    print("=" * 70)
    if args.no_url_validation:
        validation_lookup = None
    else:
        ensure_url_validation_exists(
            refresh=args.refresh_url_validation)
        validation_lookup = load_url_validation()

    outcome_files = {
        "gtr": "gtr_all_outcomes_clean.csv",
        "openalex": "openalex_all_outcomes_clean.csv",
        "scopus": "scopus_all_outcomes_clean.csv",
        "wos": "wos_all_outcomes_clean.csv"
    }

    outcome_data = {}

    for source, filename in outcome_files.items():
        outcome_file = OUTCOME_INPUT_DIR / filename

        if not outcome_file.exists():
            raise FileNotFoundError(
                f"{source} outcome dataset not found: {outcome_file}")

        outcome_data[source] = read_csv(outcome_file, OUTCOME_COLUMN_TYPES)

    outcome_map_file = OUTPUT_DIR / "project_outcome_map.csv"

    if not outcome_map_file.exists():
        raise FileNotFoundError(
            f"Project-outcome map not found: {outcome_map_file}")

    outcome_map = pd.read_csv(
        outcome_map_file,
        encoding="utf-8",
        dtype={
            "gtr_outcome_id": "string",
            "openalex_outcome_id": "string",
            "scopus_outcome_id": "string",
            "wos_outcome_id": "string"
        }
    )

    outcome_df = merge_outcomes(
        gtr_outcomes=outcome_data["gtr"],
        openalex_outcomes=outcome_data["openalex"],
        scopus_outcomes=outcome_data["scopus"],
        wos_outcomes=outcome_data["wos"],
        outcome_map=outcome_map,
        validation_lookup=validation_lookup
    )

    # Save merged outcomes
    outcome_output_file = OUTPUT_DIR / "outcomes.csv"
    outcome_df.to_csv(outcome_output_file, index=False, encoding="utf-8")
    print(f"\nSaved outcomes to: {outcome_output_file}")


if __name__ == "__main__":
    main()