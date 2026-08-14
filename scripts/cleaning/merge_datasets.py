from pathlib import Path
import argparse
import re
import subprocess
import sys
import unicodedata
import pandas as pd
from utils.col_types import PROJECT_COLUMN_TYPES, OUTCOME_COLUMN_TYPES, read_csv


# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROJECT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTCOME_INPUT_DIR = PROJECT_INPUT_DIR / "outcomes"
OUTPUT_DIR = PROJECT_INPUT_DIR / "merged"
DISAGREEMENT_DIR = OUTPUT_DIR / "disagreements"

for directory in (
    PROJECT_INPUT_DIR,
    OUTCOME_INPUT_DIR,
    OUTPUT_DIR,
    DISAGREEMENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PRIORITY = ["gtr", "scopus", "wos", "openalex"]

VALIDATION_FILE = (
    ROOT_DIR / "data" / "processed" / "unique_url_validation.csv"
)

VALIDATE_SCRIPT = Path(__file__).resolve().parent / "validate_urls.py"


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
        "domain"
    ]
    openalex_df = openalex_df[
        [col for col in openalex_cols if col in openalex_df.columns]
    ]

    # Rename OpenAlex description before merging
    if "description_clean" in openalex_df.columns:
        openalex_df = openalex_df.rename(
            columns={"description_clean": "openalex_description"}
        )

    merged_df = gtr_df.merge(
        openalex_df,
        on="project_id",
        how="left"
    )

    # Use OpenAlex funding type when GtR is missing
    if "funding_type" in merged_df.columns and "grant_category" in merged_df.columns:
        merged_df["grant_category"] = merged_df["grant_category"].fillna(
            merged_df["funding_type"]
        )
        merged_df.drop(
            columns=["funding_type"],
            inplace=True,
            errors="ignore"
        )

    # Use the longer description
    if "openalex_description" in merged_df.columns:
        merged_df["abstract_text_clean"] = merged_df.apply(
            lambda row: (
                row["openalex_description"]
                if pd.notna(row["openalex_description"])
                and len(str(row["openalex_description"])) >
                len(str(row.get("abstract_text_clean", "")))
                else row.get("abstract_text_clean")
            ),
            axis=1
        )
        merged_df.drop(
            columns=["openalex_description", "abstract_text"],
            inplace=True,
            errors="ignore"
        )

    # Remove original columns when cleaned versions exist
    cleaned_cols = [
        col for col in merged_df.columns if col.endswith("_clean")
    ]
    originals_to_remove = [
        col.replace("_clean", "")
        for col in cleaned_cols
        if col.replace("_clean", "") in merged_df.columns
    ]
    merged_df.drop(
        columns=originals_to_remove,
        inplace=True,
        errors="ignore"
    )

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
            "training": "training grant"
        })

    merged = gtr_df.merge(
        openalex_df,
        on="project_id",
        how="inner",
        suffixes=("_gtr", "_openalex")
    )

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
        record["description_difference"] = str(gtr_desc) != str(oa_desc)

        # Compare funding amounts
        gtr_funding = row.get("value_gbp_gtr")
        oa_funding = row.get("value_gbp_openalex")

        record["funding_difference"] = (
            pd.notna(gtr_funding)
            and pd.notna(oa_funding)
            and float(gtr_funding) != float(oa_funding)
        )
        record["gtr_funding"] = gtr_funding
        record["openalex_funding"] = oa_funding

        # Compare funding types
        gtr_type = row.get("grant_category")
        oa_type = row.get("funding_type")

        record["funding_type_difference"] = (
            pd.notna(gtr_type)
            and pd.notna(oa_type)
            and str(gtr_type).lower() != str(oa_type).lower()
        )
        record["gtr_grant_category"] = gtr_type
        record["openalex_funding_type"] = oa_type

        # Compare dates
        for gtr_col, oa_col, label in [
            ("start_date_gtr", "start_date_openalex", "start_date"),
            ("end_date_gtr", "end_date_openalex", "end_date")
        ]:
            gtr_date = pd.to_datetime(row.get(gtr_col), errors="coerce")
            oa_date = pd.to_datetime(row.get(oa_col), errors="coerce")

            if pd.notna(gtr_date) and pd.notna(oa_date):
                date_difference_days = (oa_date - gtr_date).days
            else:
                date_difference_days = None

            record[f"{label}_difference"] = (
                date_difference_days != 0
                if date_difference_days is not None
                else False
            )
            record[f"{label}_difference_days"] = date_difference_days
            record[f"gtr_{label}"] = gtr_date
            record[f"openalex_{label}"] = oa_date

        comparisons.append(record)

    return pd.DataFrame(comparisons)


# ---------------------------------------------------------------------------
# OUTCOME MERGE
# ---------------------------------------------------------------------------

def merge_outcomes(gtr_outcomes, openalex_outcomes, scopus_outcomes,
                             wos_outcomes, outcome_map, validation_lookup):
    """
    Build the global outcome dataset using the existing outcome map.
    """
    outcomes = outcome_map.copy()
    outcomes.drop(
        columns=["source", "match_basis", "project_id"],
        inplace=True,
        errors="ignore"
    )

    source_data = [
        ("gtr", gtr_outcomes),
        ("openalex", openalex_outcomes),
        ("scopus", scopus_outcomes),
        ("wos", wos_outcomes)
    ]

    source_id_columns = []
    sources = []

    for source, source_df in source_data:
        source_id_column = f"{source}_outcome_id"

        # Skip sources without matching IDs
        if source_id_column not in outcomes.columns:
            continue
        if "outcome_id" not in source_df.columns:
            continue

        source_df = source_df.copy()

        # Standardise outcome IDs
        outcomes[source_id_column] = (
            outcomes[source_id_column].astype("string").str.strip()
        )
        source_df["outcome_id"] = (
            source_df["outcome_id"].astype("string").str.strip()
        )

        # Keep only unique outcomes
        source_df = source_df.drop_duplicates(
            subset=["outcome_id"],
            keep="first"
        )

        source_df = source_df.rename(
            columns={
                col: f"{source}_{col}"
                for col in source_df.columns
                if col != "outcome_id"
            }
        )
        source_df = source_df.rename(
            columns={"outcome_id": source_id_column}
        )

        outcomes = outcomes.merge(
            source_df,
            on=source_id_column,
            how="left"
        )

        source_id_columns.append(source_id_column)
        sources.append(source)

    # Remove source-specific ID columns
    outcomes.drop(
        columns=source_id_columns,
        inplace=True,
        errors="ignore"
    )

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

    source_columns_to_remove = [
        f"{source}_{column}"
        for source in sources
        for column in cols_to_remove
    ]
    outcomes.drop(
        columns=source_columns_to_remove,
        inplace=True,
        errors="ignore"
    )

    # Replace original columns with cleaned versions
    clean_columns = [
        col for col in outcomes.columns
        if col.endswith("_clean") and col[:-6] in outcomes.columns
    ]

    for clean_col in clean_columns:
        original_col = clean_col[:-6]
        outcomes[original_col] = outcomes[clean_col]

    outcomes.drop(
        columns=clean_columns,
        inplace=True,
        errors="ignore"
    )

    # Merge DOI by source priority
    source_priority = ["gtr", "scopus", "wos", "openalex"]

    for column in ["doi"]:
        comparison, disagreements = check_column_agreement(
            outcomes,
            column,
            sources
        )

        source_columns = [
            f"{source}_{column}"
            for source in source_priority
            if f"{source}_{column}" in outcomes.columns
        ]

        outcomes = merge_preferred_column(
            outcomes,
            column,
            source_columns
        )

        print(f"{'Merge Rule':<30}: First populated source (GtR → Scopus → WoS → OpenAlex)")
        if disagreements.any():
            disagreement_file = DISAGREEMENT_DIR / f"{column}_disagreements.csv"
            comparison[disagreements].to_csv(disagreement_file, index=True, 
                                             encoding="utf-8")
            print(f"{'Disagreements Saved':<30}: {disagreement_file.name}")
        else:
            print(f"{'Disagreements Saved':<30}: None")

        outcomes.drop(
            columns=source_columns,
            inplace=True,
            errors="ignore"
        )

    # Merge titles using the longest value
    for column in ["title"]:
        comparison, disagreements = check_column_agreement(
            outcomes,
            column,
            sources
        )

        source_columns = [
            f"{source}_{column}"
            for source in source_priority
            if f"{source}_{column}" in outcomes.columns
        ]

        outcomes = merge_longest_column(
            outcomes,
            column,
            source_columns
        )

        print(f"{'Merge Rule':<30}: Longest non-missing {column}")
        if disagreements.any():
            disagreement_file = DISAGREEMENT_DIR / f"{column}_disagreements.csv"
            comparison[disagreements].to_csv(disagreement_file, index=True, 
                                             encoding="utf-8")
            print(f"{'Disagreements Saved':<30}: {disagreement_file.name}")
        else:
            print(f"{'Disagreements Saved':<30}: None")

        outcomes.drop(
            columns=source_columns,
            inplace=True,
            errors="ignore"
        )

    # ORGANISATION MERGE
    outcomes, organisation_disagreements = merge_organisations_by_priority(
        outcomes
    )

    if not organisation_disagreements.empty:
        organisation_disagreements.to_csv(
            DISAGREEMENT_DIR / "organisation_disagreements.csv",
            index=False,
            encoding="utf-8"
        )

        print(
            f"{'Disagreements Saved':<30}: "
            f"organisation_disagreements.csv"
        )
    else:
        print(f"{'Disagreements Saved':<30}: None")

    # URL MERGE
    final_urls = []
    url_disagreements = []

    invalid_url_count = 0
    empty_url_count = 0
    doi_fallback_count = 0
    no_url_count = 0

    for _, row in outcomes.iterrows():
        source_urls = {}

        # Collect source URLs
        for source in SOURCE_PRIORITY:
            column = f"{source}_url"

            if column not in outcomes.columns:
                source_urls[source] = ""
                continue

            value = row[column]
            source_urls[source] = (
                "" if pd.isna(value) else str(value).strip()
            )

        # Count empty and invalid source URLs
        for url in source_urls.values():
            if not url:
                empty_url_count += 1
            elif not is_usable_validation(url, validation_lookup):
                invalid_url_count += 1

        # Check for URL disagreement
        non_empty_urls = {url for url in source_urls.values() if url}

        if len(non_empty_urls) > 1:
            url_disagreements.append({
                "global_outcome_id": row["global_outcome_id"],
                **{
                    f"{source}_url": url
                    for source, url in source_urls.items()
                    if url
                }
            })

        # Select final URL
        final_url, url_source = select_final_url(
            source_urls,
            row["doi"],
            validation_lookup
        )
        final_urls.append(final_url)

        if url_source == "doi":
            doi_fallback_count += 1
        elif not final_url:
            no_url_count += 1

    outcomes["url"] = final_urls

    # Report url statistics
    print()
    print("URL Summary:")
    print("-" * 70)
    print(f"{'Empty Source URLs':<30}: {empty_url_count:,}")
    print(f"{'Invalid Source URLs':<30}: {invalid_url_count:,}")
    print(f"{'URL Disagreements':<30}: {len(url_disagreements):,}")
    print(f"{'DOI Fallbacks':<30}: {doi_fallback_count:,}")
    print(f"{'No Final URL':<30}: {no_url_count:,}")
    print(f"{'Merge Rule':<30}: First valid URL by source priority (GtR → Scopus → WoS → OpenAlex); DOI used as fallback")

    # Save url disagreements
    if url_disagreements:
        url_disagreements_df = pd.DataFrame(url_disagreements)
        url_disagreements_df.to_csv(
            DISAGREEMENT_DIR / "url_disagreements.csv",
            index=False,
            encoding="utf-8"
        )
        print(print(f"{'Disagreements Saved':<30}: url_disagreements.csv"))
    else:
        print("No URL disagreements found.")

    # Remove source url columns
    source_url_columns = [
        f"{source}_url"
        for source in sources
        if f"{source}_url" in outcomes.columns
    ]

    outcomes.drop(
        columns=source_url_columns,
        inplace=True,
        errors="ignore"
    )

    # Clean remaining columns that are only from one source
    source_columns = {}
    for column in outcomes.columns:
        for source in sources:
            prefix = f"{source}_"
            if column.startswith(prefix):
                base_column = column[len(prefix):]
                source_columns.setdefault(base_column, []).append(column)
                break
    for base_column, columns in source_columns.items():
        if len(columns) != 1:
            continue
        column = columns[0]
        outcomes.rename(columns={column: base_column}, inplace=True)

    return outcomes


# ---------------------------------------------------------------------------
# OUTCOME COLUMN MERGING
# ---------------------------------------------------------------------------

def normalise_for_comparison(value, column):
    if pd.isna(value):
        return set() if column == "organisation" else pd.NA

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKC", value)

    if column == "doi":
        value = re.sub(r"\s+", "", value)

    elif column == "title":
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"[^\w\s]", "", value)
        value = re.sub(r"\s+", " ", value).strip()

    elif column == "organisation":
        value = re.sub(r"\s*\([^)]*\)", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = {x.strip()
                 for x in value.split(";")
                 if x.strip()}

    return value


def merge_preferred_column(df, output_column, source_columns):
    """Use the first available value in source priority order."""
    df[output_column] = pd.NA

    for column in source_columns:
        if column not in df.columns:
            continue
        df[output_column] = df[output_column].fillna(df[column])

    return df


def merge_longest_column(df, output_column, source_columns):
    """Use the longest non-missing value across source columns."""
    available_columns = [
        column for column in source_columns
        if column in df.columns
    ]

    if not available_columns:
        df[output_column] = pd.NA
        return df

    lengths = df[available_columns].map(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    longest_column = lengths.idxmax(axis=1)

    df[output_column] = pd.Series(
        [
            df.loc[index, column]
            if lengths.loc[index, column] > 0
            else pd.NA
            for index, column in longest_column.items()
        ],
        index=df.index
    )

    return df

def merge_organisations_by_priority(df):
    source_columns = {
        "gtr": "gtr_organisations",
        "scopus": "scopus_institutions",
        "wos": "wos_institutions",
        "openalex": "openalex_institutions"
    }

    available_columns = [
        column
        for column in source_columns.values()
        if column in df.columns
    ]

    final_organisations = []
    disagreements = []

    records_with_organisations = 0

    for _, row in df.iterrows():
        selected_value = pd.NA
        selected_column = None

        # Select first populated source by priority
        for source in SOURCE_PRIORITY:
            column = source_columns[source]

            if column not in df.columns:
                continue

            value = row[column]

            if pd.notna(value) and str(value).strip():
                selected_value = value
                selected_column = column
                break

        final_organisations.append(selected_value)

        if selected_column is None:
            continue

        records_with_organisations += 1

        # Normalise the selected organisations
        selected_organisations = normalise_for_comparison(selected_value, "organisation")

        record = {
            "global_outcome_id": row["global_outcome_id"],
            selected_column: selected_value
        }

        disagreement = False

        # Compare every other populated source against the selected source
        for column in available_columns:
            if column == selected_column:
                continue

            value = row[column]

            if pd.isna(value) or not str(value).strip():
                continue

            record[column] = value

            other_organisations = normalise_for_comparison(value, "organisation")

            if other_organisations != selected_organisations:
                disagreement = True

        if disagreement:
            disagreements.append(record)

    df["organisations"] = final_organisations

    disagreements_df = pd.DataFrame(disagreements)

    print()
    print("Organisation Summary:")
    print("-" * 70)
    print(
        f"{'Sources Compared':<30}: "
        f"{len(available_columns):,}"
    )
    print(
        f"{'Records with Organisations':<30}: "
        f"{records_with_organisations:,}"
    )
    print(
        f"{'Records with Disagreements':<30}: "
        f"{len(disagreements_df):,}"
    )
    print(
        f"{'Merge Rule':<30}: "
        f"First populated source (GtR → Scopus → WoS → OpenAlex)"
    )

    return df, disagreements_df


def check_column_agreement(df, column, sources):
    columns = [
        f"{source}_{column}"
        for source in sources
        if f"{source}_{column}" in df.columns
    ]

    comparison = df[columns].copy()
    comparison = comparison.map(
        lambda x: normalise_for_comparison(x, column)
    )

    values_present = comparison.notna().sum(axis=1)
    disagreement = (
        (values_present >= 2)
        & (comparison.nunique(axis=1, dropna=True) > 1)
    )

    records_with_column = comparison.notna().any(axis=1).sum()

    print()
    print(f"{column.title()} Summary:")
    print("-" * 70)
    print(f"{'Sources Compared':<30}: {len(columns):,}")
    print(f"{f'Records with {column}':<30}: {records_with_column:,}")
    print(f"{'Records with Disagreements':<30}: {disagreement.sum():,}")

    return comparison, disagreement


# ---------------------------------------------------------------------------
# RUN URL VALIDATION IF NECESSARY
# ---------------------------------------------------------------------------

def ensure_url_validation_exists():
    """Run validate_urls.py if the validation cache does not exist."""
    if VALIDATION_FILE.exists():
        print()
        print("Using existing URL validation cache:")
        print(VALIDATION_FILE)
        return

    print()
    print("=" * 70)
    print("URL VALIDATION CACHE NOT FOUND")
    print("=" * 70)
    print()
    print("Running validate_urls.py...")

    if not VALIDATE_SCRIPT.exists():
        raise FileNotFoundError(
            "Could not find validate_urls.py:\n"
            f"{VALIDATE_SCRIPT}"
        )

    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT)],
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError("validate_urls.py failed.")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "validate_urls.py completed but did not create "
            "unique_url_validation.csv."
        )

    print()
    print("URL validation completed.")


# ---------------------------------------------------------------------------
# LOAD URL VALIDATION CACHE
# ---------------------------------------------------------------------------

def load_url_validation():
    """Load unique_url_validation.csv into a dictionary keyed by URL."""
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "URL validation file does not exist:\n"
            f"{VALIDATION_FILE}"
        )

    validation_df = pd.read_csv(
        VALIDATION_FILE,
        encoding="utf-8",
        dtype=str
    )

    required_columns = {"url_original", "classification"}
    missing_columns = required_columns - set(validation_df.columns)

    if missing_columns:
        raise ValueError(
            "unique_url_validation.csv is missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
        )

    validation_lookup = {}

    for _, row in validation_df.iterrows():
        url = str(row["url_original"]).strip()

        if not url:
            continue

        validation_lookup[url] = {
            "classification": str(row["classification"]).strip().lower(),
            "reason": str(row.get("reason", ""))
        }

    return validation_lookup


# ---------------------------------------------------------------------------
# URL HELPERS
# ---------------------------------------------------------------------------

def is_usable_validation(url, validation_lookup):
    """Return True when the URL is classified as valid."""
    if not url:
        return False
    result = validation_lookup.get(url)
    if result is None:
        return False
    return result["classification"] == "valid"


def select_final_url(source_urls, doi, validation_lookup):
    """Select the first valid source URL, then fall back to the DOI."""
    for source in SOURCE_PRIORITY:
        url = source_urls.get(source, "")
        if not url:
            continue
        if is_usable_validation(url, validation_lookup):
            return url, source
    doi_url = make_doi_url(doi)
    if doi_url:
        return doi_url, "doi"
    return "", ""


def make_doi_url(doi):
    """Convert a DOI identifier into an HTTPS DOI URL."""
    if pd.isna(doi):
        return ""

    doi = str(doi).strip()

    if not doi:
        return ""

    return f"https://doi.org/{doi}"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save csv comparing metadata."
    )
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
            f"OpenAlex dataset not found: {openalex_file}"
        )

    gtr_df = read_csv(gtr_file, PROJECT_COLUMN_TYPES)
    openalex_df = read_csv(gtr_file, PROJECT_COLUMN_TYPES)

    comparison_df = compare_openalex_gtr(gtr_df, openalex_df)

    if args.save_comparison:
        comparison_file = OUTPUT_DIR / "project_metadata_comparison.csv"
        comparison_df.to_csv(
            comparison_file,
            index=False,
            encoding="utf-8"
        )
        print(f"Saved comparison table as {comparison_file.name}")

    project_df = merge_projects(gtr_df, openalex_df)
    project_output_file = OUTPUT_DIR / "projects.csv"
    project_df.to_csv(
        project_output_file,
        index=False,
        encoding="utf-8"
    )

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

    print()
    print("Date Summary:")
    print("-" * 70)
    for date_label, column in [
        ("Start Date", "start_date_difference_days"),
        ("End Date", "end_date_difference_days")]:
        differences = comparison_df[column].dropna().abs()
        zero_days = (differences == 0).sum()
        one_day = (differences == 1).sum()
        more_than_one_day = (differences > 1).sum()
        print(f"\n{date_label} Differences:")
        print(f"{'Zero days off':<30}: {zero_days:,}")
        print(f"{'One day off':<30}: {one_day:,}")
        print(f"{'More than one day off':<30}: {more_than_one_day:,}")

    # ------------------------------------------------------------------
    # OUTCOMES
    # ------------------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("OUTCOME MERGE SUMMARY")
    print("=" * 70)
    ensure_url_validation_exists()
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

    # Save
    outcome_output_file = OUTPUT_DIR / "outcomes.csv"
    outcome_df.to_csv(outcome_output_file, index=False, encoding="utf-8")
    print(f"\nSaved outcomes to: {outcome_output_file}")


if __name__ == "__main__":
    main()