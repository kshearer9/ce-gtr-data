from pathlib import Path
import argparse
import re
import subprocess
import sys
import unicodedata
import pandas as pd
from utils.col_types import PROJECT_COLUMN_TYPES, OUTCOME_COLUMN_TYPES, read_csv
from utils.merge_type_map import GTR_TYPE_MAP, SCOPUS_TYPE_MAP, WOS_TYPE_MAP, OPENALEX_TYPE_MAP

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROJECT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTCOME_INPUT_DIR = PROJECT_INPUT_DIR / "outcomes"
OUTPUT_DIR = PROJECT_INPUT_DIR / "merged"
DISAGREEMENT_DIR = OUTPUT_DIR / "disagreements"
PROJECT_DISAGREEMENT_DIR = DISAGREEMENT_DIR / "projects"
OUTCOME_DISAGREEMENT_DIR = DISAGREEMENT_DIR / "outcomes"

for directory in (
    PROJECT_INPUT_DIR,
    OUTCOME_INPUT_DIR,
    OUTPUT_DIR,
    DISAGREEMENT_DIR,
    PROJECT_DISAGREEMENT_DIR,
    OUTCOME_DISAGREEMENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PRIORITY = ["gtr", "scopus", "wos", "openalex"]

VALIDATION_FILE = (
    ROOT_DIR / "data" / "processed" / "unique_url_validation.csv"
)

VALIDATE_SCRIPT = Path(__file__).resolve().parent / "validate_urls.py"

TYPE_MAPS = {
    "gtr": GTR_TYPE_MAP,
    "scopus": SCOPUS_TYPE_MAP,
    "wos": WOS_TYPE_MAP,
    "openalex": OPENALEX_TYPE_MAP,
}

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

def normalise_funding_type(value):
    """Normalise funding types for cross-source comparison."""
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    funding_type_map = {
        "research grant": "collaborative r&d",
        "feasibility studies": "research grant",
        "eu-funded": "grant"
    }
    return funding_type_map.get(value, value)

def compare_projects(gtr_df, openalex_df):
    """Compare GtR and OpenAlex project metadata and report useful statistics."""

    merged = gtr_df.merge(
        openalex_df,
        on="project_id",
        how="inner",
        suffixes=("_gtr", "_openalex")
    )

    disagreements = {}

    print_summary_header("Project Source Coverage")
    print(f"{'Records in GtR':<35}: {len(gtr_df):,}")
    print(f"{'Records in OpenAlex':<35}: {len(openalex_df):,}")
    print(f"{'Records in both sources':<35}: {len(merged):,}")

    # Description
    description_records = []

    gtr_desc = merged.get("abstract_text_clean")
    oa_desc = merged.get("description_clean")

    if gtr_desc is not None and oa_desc is not None:
        gtr_present = gtr_desc.notna() & gtr_desc.astype(str).str.strip().ne("")
        oa_present = oa_desc.notna() & oa_desc.astype(str).str.strip().ne("")
        both_present = gtr_present & oa_present

        agreement = pd.Series(False, index=merged.index)

        for index in merged.index[both_present]:
            gtr_normalised = normalise_for_comparison(
                gtr_desc.loc[index], "description"
            )
            oa_normalised = normalise_for_comparison(
                oa_desc.loc[index], "description"
            )

            if gtr_normalised == oa_normalised:
                agreement.loc[index] = True
            else:
                description_records.append({
                    "project_id": merged.loc[index, "project_id"],
                    "gtr_description": gtr_desc.loc[index],
                    "openalex_description": oa_desc.loc[index],
                    "gtr_description_length": len(str(gtr_desc.loc[index])),
                    "openalex_description_length": len(
                        str(oa_desc.loc[index])
                    )
                })

        description_disagreements = pd.DataFrame(description_records)

        print_summary_header("Project Description")
        print(f"{'GtR populated':<35}: {gtr_present.sum():,}")
        print(f"{'OpenAlex populated':<35}: {oa_present.sum():,}")
        print(f"{'Agree after normalisation':<35}: {agreement.sum():,}")
        print(
            f"{'Disagree':<35}: "
            f"{(both_present & ~agreement).sum():,}"
        )

        if not description_disagreements.empty:
            gtr_lengths = description_disagreements[
                "gtr_description_length"
            ]
            oa_lengths = description_disagreements[
                "openalex_description_length"
            ]

            print(
                f"{'OpenAlex longer':<35}: "
                f"{(oa_lengths > gtr_lengths).sum():,}"
            )
            print(
                f"{'GtR longer':<35}: "
                f"{(gtr_lengths > oa_lengths).sum():,}"
            )

        disagreements["description"] = description_disagreements

    else:
        disagreements["description"] = pd.DataFrame()

    # Funding
    funding_records = []

    gtr_funding = merged.get("value_gbp_gtr")
    oa_funding = merged.get("value_gbp_openalex")

    if gtr_funding is not None and oa_funding is not None:
        gtr_funding = pd.to_numeric(gtr_funding, errors="coerce")
        oa_funding = pd.to_numeric(oa_funding, errors="coerce")

        both_present = gtr_funding.notna() & oa_funding.notna()

        difference = (oa_funding - gtr_funding).abs()
        exact = both_present & difference.eq(0)
        small = both_present & (difference > 0) & (difference <= 1)
        medium = both_present & (difference > 1) & (difference <= 1000)
        large = both_present & (difference > 1000)

        for index in merged.index[both_present & ~exact]:
            funding_records.append({
                "project_id": merged.loc[index, "project_id"],
                "gtr_funding": gtr_funding.loc[index],
                "openalex_funding": oa_funding.loc[index],
                "difference_gbp": (
                    oa_funding.loc[index] - gtr_funding.loc[index]
                )
            })

        funding_disagreements = pd.DataFrame(funding_records)

        print_summary_header("Project Funding")
        print(f"{'GtR populated':<35}: {gtr_funding.notna().sum():,}")
        print(f"{'OpenAlex populated':<35}: {oa_funding.notna().sum():,}")
        print(f"{'Exact agreement':<35}: {exact.sum():,}")
        print(f"{'Difference ≤ £1':<35}: {small.sum():,}")
        print(f"{'Difference £1–£1,000':<35}: {medium.sum():,}")
        print(f"{'Difference > £1,000':<35}: {large.sum():,}")

        openalex_higher = both_present & (oa_funding > gtr_funding)
        gtr_higher = both_present & (gtr_funding > oa_funding)

        print(f"{'OpenAlex higher':<35}: {openalex_higher.sum():,}")
        print(f"{'GtR higher':<35}: {gtr_higher.sum():,}")

        disagreements["funding"] = funding_disagreements

    else:
        disagreements["funding"] = pd.DataFrame()

    # Funding type
    funding_type_records = []

    if (
        "grant_category" in merged.columns
        and "funding_type" in merged.columns
    ):
        merged["funding_type"] = merged["funding_type"].replace({
            "research": "research grant",
            "voucher": "vouchers",
            "training": "training grant"
        })

        gtr_type = merged["grant_category"]
        oa_type = merged["funding_type"]

        both_present = (
            gtr_type.notna()
            & oa_type.notna()
            & gtr_type.astype(str).str.strip().ne("")
            & oa_type.astype(str).str.strip().ne("")
        )

        agreement = pd.Series(False, index=merged.index)

        for index in merged.index[both_present]:
            gtr_normalised = normalise_funding_type(
                gtr_type.loc[index]
            )
            oa_normalised = normalise_funding_type(
                oa_type.loc[index]
            )

            agreement.loc[index] = (
                gtr_normalised == oa_normalised
            )

            if not agreement.loc[index]:
                funding_type_records.append({
                    "project_id": merged.loc[index, "project_id"],
                    "gtr_grant_category": gtr_type.loc[index],
                    "openalex_funding_type": oa_type.loc[index]
                })

        funding_type_disagreements = pd.DataFrame(
            funding_type_records
        )

        print_summary_header("Project Funding Type")
        print(f"{'GtR populated':<35}: {gtr_type.notna().sum():,}")
        print(f"{'OpenAlex populated':<35}: {oa_type.notna().sum():,}")
        print(f"{'Agree after mapping':<35}: {agreement.sum():,}")
        print(
            f"{'Disagree':<35}: "
            f"{(both_present & ~agreement).sum():,}"
        )

        disagreements["funding_type"] = funding_type_disagreements

    else:
        disagreements["funding_type"] = pd.DataFrame()

    # DATES
    for column in ["start_date", "end_date"]:

        gtr_column = f"{column}_gtr"
        oa_column = f"{column}_openalex"

        if (
            gtr_column not in merged.columns
            or oa_column not in merged.columns
        ):
            disagreements[column] = pd.DataFrame()
            continue

        gtr_dates = pd.to_datetime(
            merged[gtr_column],
            errors="coerce"
        )
        oa_dates = pd.to_datetime(
            merged[oa_column],
            errors="coerce"
        )

        both_present = gtr_dates.notna() & oa_dates.notna()

        difference_days = (
            oa_dates - gtr_dates
        ).dt.days

        absolute_difference = difference_days.abs()

        exact = both_present & absolute_difference.eq(0)
        one_day = both_present & absolute_difference.eq(1)
        more_than_one_day = (
            both_present & absolute_difference.gt(1)
        )

        date_records = []

        for index in merged.index[more_than_one_day | one_day]:
            date_records.append({
                "project_id": merged.loc[index, "project_id"],
                f"gtr_{column}": gtr_dates.loc[index],
                f"openalex_{column}": oa_dates.loc[index],
                "difference_days": difference_days.loc[index]
            })

        date_disagreements = pd.DataFrame(date_records)

        print_summary_header(
            f"Project {column.replace('_', ' ').title()}"
        )
        print(
            f"{'GtR populated':<35}: "
            f"{gtr_dates.notna().sum():,}"
        )
        print(
            f"{'OpenAlex populated':<35}: "
            f"{oa_dates.notna().sum():,}"
        )
        print(f"{'Exact agreement':<35}: {exact.sum():,}")
        print(f"{'Off by 1 day':<35}: {one_day.sum():,}")
        print(
            f"{'Off by >1 day':<35}: "
            f"{more_than_one_day.sum():,}"
        )

        earlier = both_present & difference_days.lt(0)
        later = both_present & difference_days.gt(0)

        print(
            f"{'OpenAlex earlier':<35}: "
            f"{earlier.sum():,}"
        )
        print(
            f"{'OpenAlex later':<35}: "
            f"{later.sum():,}"
        )

        disagreements[column] = date_disagreements

    return disagreements


# ---------------------------------------------------------------------------
# OUTCOME COLUMN MERGING
# ---------------------------------------------------------------------------

def get_source_columns(df, sources, column):
    return [
        f"{source}_{column}"
        for source in sources
        if f"{source}_{column}" in df.columns
    ]

def normalise_for_comparison(value, column):
    if pd.isna(value):
        return set() if column == "organisation" else pd.NA

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKC", value)

    if column == "doi":
        value = re.sub(r"\s+", "", value)

    elif column in ["source_title", "journal"]:
        # Treat "&" and "and" as equivalent.
        value = re.sub(r"\s*&\s*", " and ", value)
        # Remove brackets
        value = re.sub(r"\([^)]*\)", " ", value)
        # Remove everything from the first ":" onwards.
        value = re.sub(r":.*$", "", value)
        # Remove everything from the first "-" onwards.
        value = re.sub(r"-.*$", "", value)
        # Remove remaining punctuation.
        value = re.sub(r"[^\w\s]", " ", value)
        # Normalise whitespace.
        value = re.sub(r"\s+", " ", value).strip()

    elif column in ["title", "abstract", "description"]:
        # Normalise whitespace
        value = re.sub(r"\s+", " ", value).strip()
        # Remove punctuation for comparison
        value = re.sub(r"[^\w\s]", "", value)
        # Normalise whitespace again
        value = re.sub(r"\s+", " ", value).strip()

    elif column == "organisation":
        value = re.sub(r"\s*\([^)]*\)", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = {
            x.strip()
            for x in value.split(";")
            if x.strip()
        }

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
            selected_column: selected_value}
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

    print_summary_header("Organisation Summary:")
    print(f"{'Sources Compared':<35}: {len(available_columns):,}")
    print(f"{'Records with Organisations':<35}: {records_with_organisations:,}")
    print(f"{'Records with Disagreements':<35}: {len(disagreements_df):,}")
    print(f"{'Merge Rule':<35}: First populated source (GtR → Scopus → WoS → OpenAlex)")
    df.drop(columns=available_columns, inplace=True, errors="ignore")
    return df, disagreements_df


def merge_title_and_abstract(df):
    """
    Merge source abstracts into the final abstract column.

    Abstract columns are detected dynamically as any column ending in
    '_abstract', excluding gtr_abstract.

    The longest non-empty abstract is selected.

    Titles are assumed to have already been generically merged into
    the 'title' column. Only missing titles are filled using the
    GtR description, provided the description is not the same as
    the selected abstract.
    """
    # Find all source abstract columns except GtR abstract.
    abstract_columns = [
        column for column in df.columns
        if column.endswith("_abstract") and column != "gtr_abstract"
    ]
    # Find the GtR description column.
    gtr_description_column = (
        "gtr_description" if "gtr_description" in df.columns else None
    )

    final_abstracts = []
    source_abstract_count = 0
    gtr_fallback_count = 0
    no_abstract_count = 0
    # Select the longest available source abstract.
    for _, row in df.iterrows():
        abstracts = []
        for column in abstract_columns:
            value = clean_text_value(row.get(column))
            if value:
                abstracts.append(value)
        if abstracts:
            selected_abstract = max(abstracts, key=len)
            source_abstract_count += 1
        else:
            selected_abstract = ""

        # Use GtR description when no source abstract is available.
        if not selected_abstract and gtr_description_column:
            selected_abstract = clean_text_value(
                row.get(gtr_description_column))
            if selected_abstract:
                gtr_fallback_count += 1
        if not selected_abstract:
            no_abstract_count += 1
        final_abstracts.append(
            selected_abstract if selected_abstract else pd.NA)

    # Add the final abstract after the loop.
    df["abstract"] = final_abstracts

    # Fill missing titles
    existing_title_count = 0
    title_filled_count = 0
    title_match_abstract_count = 0
    missing_title_count = 0
    if gtr_description_column:
        for index, row in df.iterrows():
            existing_title = clean_text_value(row.get("title"))
            if existing_title:
                existing_title_count += 1
                continue
            missing_title_count += 1
            gtr_description = clean_text_value(row.get(gtr_description_column))
            if not gtr_description:
                continue
            selected_abstract = clean_text_value(row.get("abstract"))

            # Do not use the description as a title if it is the abstract.
            if (selected_abstract
                and normalise_for_comparison(
                    gtr_description, "description"
                ) == normalise_for_comparison(
                    selected_abstract, "abstract")):
                title_match_abstract_count += 1
                continue
            df.at[index, "title"] = gtr_description
            title_filled_count += 1
    final_abstract_count = df["abstract"].notna().sum()
    final_title_count = df["title"].notna().sum()

    print_summary_header("Description, Abstract and Title Summary:")
    print(f"{'Abstract from source':<35}: {source_abstract_count:,}")
    print(f"{'GtR description used as abstract':<35}: {gtr_fallback_count:,}")
    print(f"{'No abstract available':<35}: {no_abstract_count:,}")
    print(f"{'Final abstracts':<35}: {final_abstract_count:,}")
    print()
    print(f"{'Existing titles':<35}: {existing_title_count:,}")
    print(f"{'Missing titles after source merge':<35}: "
          f"{missing_title_count:,}")
    print(f"{'Titles filled from GtR description':<35}: "
          f"{title_filled_count:,}")
    print(f"{'Description matched abstract':<35}: "
          f"{title_match_abstract_count:,}")
    print(f"{'Final titles':<35}: {final_title_count:,}")
    print(f"{'Merge Rule':<35}: "
          "Longest non-missing abstract; GtR description as fallback")
    
    # Remove original columns after merging
    if gtr_description_column:
        df.drop(columns=[gtr_description_column],
                inplace=True, errors="ignore")
    if abstract_columns:
        df.drop(columns=abstract_columns,
                inplace=True, errors="ignore")
    return df


def merge_source_title_and_journal(df, sources=None):
    """
    Merge source titles in priority order:
        WoS → Scopus → OpenAlex → GtR

    Then fill any remaining missing source titles from the final
    journal column.
    """
    if sources is None:
        sources = ["wos", "scopus"]
    # Find source title columns
    source_title_columns = get_source_columns(df, sources, "source_title")
    # Find journal columns
    journal_columns = get_source_columns(df, sources, "journal")
    df["source_title"] = pd.NA

    # Count how many records are filled from each source.
    source_fill_counts = {}
    for source in sources:
        column = f"{source}_source_title"
        if column not in df.columns:
            source_fill_counts[source] = 0
            continue
        missing_source_title = (
            df["source_title"].isna()
            | df["source_title"].astype("string").str.strip().eq(""))
        populated = (
            df[column].notna()
            & df[column].astype("string").str.strip().ne(""))
        fill_mask = missing_source_title & populated
        df.loc[fill_mask, "source_title"] = df.loc[
            fill_mask, column]
        source_fill_counts[source] = int(fill_mask.sum())

    # If a final journal column already exists, use it.
    # Otherwise construct it from the source-specific journal columns
    # using the same priority order.
    if "journal" not in df.columns:
        df["journal"] = pd.NA
    journal_fill_counts = {}
    for source in sources:
        column = f"{source}_journal"
        if column not in df.columns:
            journal_fill_counts[source] = 0
            continue
        missing_journal = (df["journal"].isna()
                           | df["journal"].astype("string").str.strip().eq(""))
        populated = (df[column].notna()
                     & df[column].astype("string").str.strip().ne(""))
        fill_mask = missing_journal & populated
        df.loc[fill_mask, "journal"] = df.loc[fill_mask, column]
        journal_fill_counts[source] = int(fill_mask.sum())

    # Use journal as final source title fallback
    source_title_missing = (
        df["source_title"].isna()
        | df["source_title"].astype("string").str.strip().eq(""))
    journal_populated = (
        df["journal"].notna()
        & df["journal"].astype("string").str.strip().ne(""))
    journal_fallback_mask = (source_title_missing & journal_populated)
    journal_fallback_count = int(journal_fallback_mask.sum())
    df.loc[journal_fallback_mask, 
           "source_title"] = df.loc[journal_fallback_mask,"journal"]

    source_title_disagreements = pd.DataFrame()
    source_title_values = df["source_title"].map(
        lambda x: normalise_for_comparison(x, "source_title"))
    journal_values = df["journal"].map(
        lambda x: normalise_for_comparison(x, "journal"))
    source_title_present = source_title_values.notna()
    journal_present = journal_values.notna()
    comparable = source_title_present & journal_present
    disagreement = (comparable & (source_title_values != journal_values))
    if disagreement.any():
        disagreement_columns = ["source_title", "journal"]
        if "global_outcome_id" in df.columns:
            source_title_disagreements = df.loc[
                disagreement,
                ["global_outcome_id"] + disagreement_columns
            ].copy()
        else:
            source_title_disagreements = df.loc[
                disagreement,
                disagreement_columns
            ].copy()
    source_title_journal_disagreements = int(disagreement.sum())
    final_source_titles = (
        df["source_title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum())

    print_summary_header("Source Title / Journal Summary:")
    print(f"{'Source title columns merged':<35}: "
          f"{len(source_title_columns):,}")
    print(f"{'Journal columns merged':<35}: {len(journal_columns):,}")
    for source in sources:
        count = source_fill_counts.get(source, 0)
        if count > 0:
            print(f"{'Filled source title from ' + source:<35}: "
                  f"{count:,}")
    for source in sources:
        count = journal_fill_counts.get(source, 0)
        if count > 0:
            print(f"{'Fallback source title from ' + source:<35}: "
                  f"{journal_fallback_count:,}")
    print(f"{'Final source titles':<35}: {final_source_titles:,}")
    print(f"{'Source title disagreements':<35}: "
          f"{source_title_journal_disagreements:,}")
    if not source_title_disagreements.empty:
        disagreement_file = OUTCOME_DISAGREEMENT_DIR / "source_title_disagreements.csv"
        source_title_disagreements.to_csv(
            disagreement_file, index=False, encoding="utf-8")
        print(f"{'Disagreements Saved':<35}: {disagreement_file.name}")
    print(
        f"{'Merge Rule':<35}: "
        "First populated source (WoS → Scopus)")

    df.drop(
        columns=source_title_columns + journal_columns,
        inplace=True, errors="ignore")
    return df

def merge_majority_column(df, column, sources):
    source_columns = get_source_columns(df, sources, column)
    if not source_columns:
        print(f"No source columns found for {column}.")
        return df
    final_values = []
    selected_sources = []
    for _, row in df.iterrows():
        # Get available values in source priority order
        values = {
            source: row[f"{source}_{column}"]
            for source in sources
            if f"{source}_{column}" in df.columns
            and pd.notna(row[f"{source}_{column}"])}
        if not values:
            final_values.append(pd.NA)
            selected_sources.append(None)
            continue
        # Count how many sources report each value
        value_counts = pd.Series(values.values()).value_counts()
        # Check for a majority
        majority_values = value_counts[
            value_counts > len(values) / 2]
        if not majority_values.empty:
            # Majority value wins
            selected_value = majority_values.index[0]
            # Find which priority source supplied that value
            selected_source = next(
                source
                for source in sources
                if source in values
                and values[source] == selected_value)
        else:
            # No majority: use source priority
            selected_source = next(iter(values))
            selected_value = values[selected_source]
        final_values.append(selected_value)
        selected_sources.append(selected_source)
    df[column] = final_values
    # Report which source supplied the final value
    for source in sources:
        count = sum(
            selected_source == source
            for selected_source in selected_sources)
        print(f"{source.title() + ' selected':<35}: "
              f"{count:,}")
    return df

def map_outcome_types(df, source, type_map):
    """
    Map raw source types to unified type/subtype values.
    Missing values -> ('other', 'other')
    Unknown values -> ('other', 'other') with a warning.
    """
    raw_column = f"{source}_type"
    if raw_column not in df.columns:
        return df, set()
    unmapped = set()
    mapped_types = []
    mapped_subtypes = []
    for value in df[raw_column]:
        if pd.isna(value) or not str(value).strip():
            mapped_types.append(pd.NA)
            mapped_subtypes.append(pd.NA)
            continue
        raw_value = str(value).strip().lower()
        if raw_value not in type_map:
            unmapped.add(raw_value)
            mapped_types.append("other")
            mapped_subtypes.append("other")
            continue
        mapped_type, mapped_subtype = type_map[raw_value]
        mapped_types.append(mapped_type)
        mapped_subtypes.append(mapped_subtype)
    df[f"{source}_type"] = mapped_types
    df[f"{source}_subtype"] = mapped_subtypes
    if unmapped:
        print(f"WARNING: {source} has {len(unmapped)} "
             f"unmapped type categor{'y' if len(unmapped) == 1 else 'ies'}:")
        for category in sorted(unmapped):
            print(f"  - {category}")
    return df, unmapped

def merge_outcome_types(df, sources=None):
    """
    Merge mapped outcome types using majority vote, with source priority
    as the tie-breaker.

    Rules:
    - Blank values are ignored.
    - ('other', 'other') is ignored when looking for a useful value.
    - Majority vote is used among useful values.
    - If there is no majority, source priority determines the result.
    - If every source is blank or ('other', 'other'), final value is
      ('other', 'other').
    """
    if sources is None:
        sources = SOURCE_PRIORITY
    final_types = []
    final_subtypes = []
    for _, row in df.iterrows():
        values = {}
        for source in sources:
            type_column = f"{source}_type"
            subtype_column = f"{source}_subtype"
            if type_column not in df.columns:
                continue
            source_type = row[type_column]
            if pd.isna(source_type) or not str(source_type).strip():
                continue
            source_type = str(source_type).strip().lower()
            source_subtype = row.get(subtype_column)
            if (pd.isna(source_subtype)
                or not str(source_subtype).strip()):
                source_subtype = "other"
            else:
                source_subtype = str(source_subtype).strip().lower()
            # Ignore "other / other" during the vote
            if (source_type == "other"
                and source_subtype == "other"):
                continue
            values[source] = (source_type, source_subtype)

        # No useful values anywhere
        if not values:
            final_types.append("other")
            final_subtypes.append("other")
            continue

        # Count complete (type, subtype) pairs
        value_counts = pd.Series(
            list(values.values()),
            dtype="object"
        ).value_counts()

        # Majority requires > 50% of useful sources
        majority_values = value_counts[
            value_counts > len(values) / 2]

        if not majority_values.empty:
            selected_value = majority_values.index[0]

        else:
            # No majority: use source priority
            selected_source = next(
                source
                for source in sources
                if source in values)
            selected_value = values[selected_source]
        selected_type, selected_subtype = selected_value
        final_types.append(selected_type)
        final_subtypes.append(selected_subtype)

    df["type"] = final_types
    df["subtype"] = final_subtypes

    # Final GtR fallback for records with no type/subtype
    fallback_unmapped_values = {}
    if "gtr_gtr_outcome_type" in df.columns:

        fallback_count = 0
        fallback_unmapped_count = 0
        fallback_missing_count = 0

        for index, row in df.iterrows():

            current_type = row["type"]
            current_subtype = row["subtype"]

            type_missing = (
                pd.isna(current_type)
                or not str(current_type).strip()
                or str(current_type).strip().lower() == "other"
            )

            subtype_missing = (
                pd.isna(current_subtype)
                or not str(current_subtype).strip()
                or str(current_subtype).strip().lower() == "other"
            )

            # Only fallback when BOTH are missing/useless
            if not (type_missing and subtype_missing):
                continue

            gtr_value = row["gtr_gtr_outcome_type"]

            if pd.isna(gtr_value) or not str(gtr_value).strip():
                fallback_missing_count += 1
                continue

            gtr_value = str(gtr_value).strip().lower()

            if gtr_value not in GTR_TYPE_MAP:
                fallback_unmapped_count += 1
                fallback_unmapped_values[gtr_value] = (
                    fallback_unmapped_values.get(gtr_value, 0) + 1
                )
                continue

            mapped_type, mapped_subtype = GTR_TYPE_MAP[gtr_value]

            # Only replace other/other with a useful GtR mapping
            if mapped_type != "other" or mapped_subtype != "other":
                df.at[index, "type"] = mapped_type
                df.at[index, "subtype"] = mapped_subtype
                fallback_count += 1
    return df


def check_column_agreement(df, column, sources, fallback_columns = None):
    columns = get_source_columns(df, sources, column)
    if not columns:
        print_summary_header(f"{column.title()} Summary:")
        print(f"No source columns found for {column}.")
        return pd.DataFrame(index=df.index), pd.Series(
            False, index=df.index)
    if "global_outcome_id" in df.columns:
        comparison_columns = ["global_outcome_id"] + columns
    else:
        comparison_columns = columns
    comparison = df[comparison_columns].copy()
    source_comparison = comparison[columns].map(
        lambda x: normalise_for_comparison(x, column))
    values_present = source_comparison.notna().sum(axis=1)
    disagreement = (
        (values_present >= 2)
        & (source_comparison.nunique(axis=1, dropna=True) > 1))

    # Records with a value in the normal source columns
    records_with_column = source_comparison.notna().any(axis=1)

    # Include fallback columns in coverage count
    fallback_present = pd.Series(False, index=df.index)
    if fallback_columns:
        for fallback_column in fallback_columns:
            if fallback_column not in df.columns:
                continue
            fallback_values = df[fallback_column].map(
                lambda x: normalise_for_comparison(x, column))
            fallback_present |= fallback_values.notna()
    records_with_column_or_fallback = (
        records_with_column | fallback_present)

    # Summary
    print_summary_header(f"{column.title()} Summary:")
    print(f"{'Sources Compared':<35}: {len(columns):,}")
    print(f"{f'Records with {column}':<35}: {records_with_column.sum():,}")
    if fallback_columns:
        print(f"{f'Records with {column} after fallback':<35}: "
              f"{records_with_column_or_fallback.sum():,}")
        for fallback_column in fallback_columns:
            if fallback_column in df.columns:
                fallback_values = df[fallback_column].map(
                    lambda x: normalise_for_comparison(x, column))
    print(f"{'Records with Disagreements':<35}: {disagreement.sum():,}")
    return comparison, disagreement


def save_disagreements(disagreements, comparison, column, dataset):
    """Save comparison rows where source values disagree."""
    if not disagreements.any():
        print(f"{'Disagreements Saved':<35}: None")
        return
    if dataset == "projects":
        disagreement_dir = PROJECT_DISAGREEMENT_DIR
    elif dataset == "outcomes":
        disagreement_dir = OUTCOME_DISAGREEMENT_DIR
    else:
        raise ValueError(f"Invalid dataset '{dataset}'. "
                         "Expected 'project' or 'outcome'.")
    disagreement_dir.mkdir(parents=True, exist_ok=True)
    disagreement_file = disagreement_dir / f"{column}_disagreements.csv"
    output = comparison.loc[disagreements].copy()
    # Add global outcome ID if available
    if "global_outcome_id" in comparison.columns:
        output = output[["global_outcome_id"]
                        + [col for col in output.columns
                           if col != "global_outcome_id"]]
    elif "global_outcome_id" in comparison.index.names:
        output = output.reset_index()
    output.to_csv(disagreement_file, index=False, encoding="utf-8")
    print(f"{'Disagreements Saved':<35}: {disagreement_file.name}")


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

def fill_wos_issns_from_scopus(df):
    """
    Fill missing WoS ISSN/eISSN values from Scopus and report ISSN
    agreement/disagreement statistics.

    Disagreement rules:
    - Skip if Scopus is completely empty.
    - Skip if WoS ISSN and WoS eISSN are both completely empty.
    - One Scopus ISSN agrees if it matches either WoS ISSN/eISSN.
    - Multiple Scopus ISSNs agree only if every Scopus ISSN matches
      either WoS ISSN/eISSN.
    """
    if "scopus_issn" not in df.columns:
        return df
    if "wos_issn" not in df.columns:
        df["wos_issn"] = pd.NA
    if "wos_eissn" not in df.columns:
        df["wos_eissn"] = pd.NA

    filled_issn = 0
    filled_eissn = 0
    skipped_single = 0
    skipped_ambiguous = 0
    disagreement_records = []
    for index, row in df.iterrows():
        wos_issn = (
            str(row["wos_issn"]).strip()
            if pd.notna(row["wos_issn"]) else "")
        wos_eissn = (
            str(row["wos_eissn"]).strip()
            if pd.notna(row["wos_eissn"]) else "")
        scopus_value = row["scopus_issn"]

        # skip rows where Scopus is empty
        if pd.isna(scopus_value) or not str(scopus_value).strip():
            continue
        scopus_values = {
            value.strip()
            for value in str(scopus_value).split(";")
            if value.strip()}
        if not scopus_values:
            continue
        wos_values = {
            value for value in [wos_issn, wos_eissn] if value}

        # check disagreement before filling WoS
        if wos_values:
            matching_issns = scopus_values & wos_values
            if not matching_issns:
                disagreement_records.append({
                    "global_outcome_id": row["global_outcome_id"],
                    "wos_issn": row["wos_issn"],
                    "wos_eissn": row["wos_eissn"],
                    "scopus_issn": row["scopus_issn"]
                })

        # fill the missing WoS value when there is only one candidate
        if wos_issn and not wos_eissn:
            remaining = scopus_values - {wos_issn}
            if len(remaining) == 1:
                df.at[index, "wos_eissn"] = remaining.pop()
                filled_eissn += 1
        elif wos_eissn and not wos_issn:
            remaining = scopus_values - {wos_eissn}
            if len(remaining) == 1:
                df.at[index, "wos_issn"] = remaining.pop()
                filled_issn += 1

        # do not guess when both WoS values are missing
        elif not wos_issn and not wos_eissn:
            if len(scopus_values) == 1:
                skipped_single += 1
            elif len(scopus_values) == 2:
                skipped_ambiguous += 1

    records_with_issns = (
        df["wos_issn"].fillna("").astype(str).str.strip().ne("").sum())
    records_with_eissns = (
        df["wos_eissn"].fillna("").astype(str).str.strip().ne("").sum())

    disagreements_df = pd.DataFrame(disagreement_records)
    print_summary_header("Scopus → WoS ISSN Summary:")
    print(f"{'WoS ISSNs filled':<35}: {filled_issn:,}")
    print(f"{'WoS eISSNs filled':<35}: {filled_eissn:,}")
    print(f"{'Single Scopus ISSN skipped':<35}: {skipped_single:,}")
    print(f"{'Ambiguous Scopus pairs skipped':<35}: {skipped_ambiguous:,}")
    print(f"{'Records with ISSNs':<35}: {records_with_issns:,}")
    print(f"{'Records with eISSNs':<35}: {records_with_eissns:,}")
    print(f"{'Records with Disagreements':<35}: {len(disagreements_df):,}")
    print(f"{'Merge Rule':<35}: Scopus ISSN(s) must match WoS ISSN/eISSN; records with either source empty are skipped")

    if not disagreements_df.empty:
        disagreement_file = DISAGREEMENT_DIR / "issn_disagreements.csv"
        disagreements_df.to_csv(
            disagreement_file, index=False, encoding="utf-8")
        print(f"{'Disagreements Saved':<35}: "
              f"{disagreement_file.name}")
        
    df.drop(columns=["scopus_issn"], inplace=True, errors="ignore")
    return df

def print_summary_header(title):
    title = title.replace("_", " ").title()
    print()
    print(title)
    print("-" * 70)

def keyword_summary(df):
    """Report keyword coverage for source keyword columns and author keywords."""
    # Normal keyword columns: *_keywords, but NOT *_author_keywords
    keyword_columns = [
        column for column in df.columns
        if column.endswith("_keywords")
        and not column.endswith("_author_keywords")
    ]
    # Author keyword columns
    author_keyword_columns = [
        column for column in df.columns
        if column.endswith("_author_keywords")
    ]
    print_summary_header("Keyword Summary:")
    if not keyword_columns and not author_keyword_columns:
        print("No *_keywords or *_author_keywords columns found.")
        return
    any_keywords = pd.Series(False, index=df.index)

    # Normal keywords
    for column in keyword_columns:
        values = df[column].fillna("").astype(str).str.strip()
        populated = values.ne("")
        # Include normal keywords in overall coverage
        any_keywords |= populated
        keyword_count = values.apply(
            lambda x: sum(
                1 for keyword in x.split(";")
                if keyword.strip())).sum()
        print(f"{column}:")
        print(f"  Rows with >=1 keyword : {populated.sum():,}")
        print(f"  Total keyword entries : {keyword_count:,}")
    if len(keyword_columns) > 1:
        print()
        print("OVERALL KEYWORDS:")
        print(f"  Rows with >=1 keyword : {any_keywords.sum():,}")
        print(f"  Rows with no keywords : {(~any_keywords).sum():,}")

    # Author keywords
    if author_keyword_columns:
        print_summary_header("Author Keywords:")
        for column in author_keyword_columns:
            values = df[column].fillna("").astype(str).str.strip()
            populated = values.ne("")
            keyword_count = values.apply(
                lambda x: sum(
                    1 for keyword in x.split(";")
                    if keyword.strip())).sum()
            print(f"{column}:")
            print(f"  Rows with >=1 keyword : {populated.sum():,}")
            print(f"  Total keyword entries : {keyword_count:,}")

def merge_count(df, column, sources):
    source_columns = get_source_columns(df, sources, column)
    if not source_columns:
        print(f"No source columns found for {column}.")
        return df
    comparison, disagreements = check_column_agreement(
        df, column, sources)
    # Coverage by source
    for source in sources:
        source_column = f"{source}_{column}"
        if source_column in df.columns:
            present = df[source_column].notna()
            print(f"{source.title() + ' counts':<35}: "
                  f"{present.sum():,}")
    # Difference statistics
    if len(source_columns) >= 2:
        first = source_columns[0]
        second = source_columns[1]
        both_present = (
            df[first].notna()
            & df[second].notna())
        difference = (df[first] - df[second]).abs()
        exact_match = (both_present & (difference == 0))
        less_than_3 = (both_present & (difference > 0)
                       & (difference < 3))
        three_or_more = (both_present & (difference >= 3))
        print(f"{'Both sources present':<35}: {both_present.sum():,}")
        print(f"{'Exact match':<35}: {exact_match.sum():,}")
        print(f"{'Less than 3 difference':<35}: {less_than_3.sum():,}")
        print(f"{'3+ difference':<35}: {three_or_more.sum():,}")
    df = merge_preferred_column(df, column, source_columns)
    print(f"{'Merge Rule':<35}: First populated source "
          f"({' → '.join(source.title() for source in sources)})")
    save_disagreements(disagreements, comparison, column, "outcomes")  
    df.drop(columns=source_columns, inplace=True, errors="ignore")
    return df

def get_source_priority_by_outliers(df, column, sources):
    source_columns = get_source_columns(df, sources, column)
    outlier_counts = {source: 0 for source in sources}
    for _, row in df[source_columns].iterrows():
        present = row.dropna()
        if len(present) < 3:
            continue
        counts = present.value_counts()
        # One source differs while the others agree
        if len(counts) == 2 and counts.iloc[-1] == 1:
            outlier_column = present[
                present == counts.index[-1]
            ].index[0]
            source = outlier_column.removesuffix(f"_{column}")
            if source in outlier_counts:
                outlier_counts[source] += 1
    priority = sorted(sources,
                      key=lambda source: outlier_counts[source])
    for source in priority:
        print(f"  {source.title() + ' outlier':<33}: "
              f"{outlier_counts[source]:,}")
    return priority



# ---------------------------------------------------------------------------
# OUTCOME MERGE
# ---------------------------------------------------------------------------

def clean_text_value(value):
    """Return a cleaned string, or an empty string for missing values."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""
    return re.sub(r"\s+", " ", value)
      
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

    # TITLE MERGE
    for column in ["title"]:
        title_comparison, title_disagreements = check_column_agreement(
            outcomes, column, sources)
        source_columns = get_source_columns(outcomes, SOURCE_PRIORITY, column)
        outcomes = merge_longest_column(outcomes, column, source_columns)
        print(f"{'Merge Rule':<35}: Longest non-missing {column}")
        save_disagreements(title_disagreements, title_comparison, column, "outcomes")
        outcomes.drop(columns=source_columns, inplace=True, errors="ignore")

    # TYPE MERGE
    for source in sources:
        outcomes, _ = map_outcome_types(
            outcomes, source, TYPE_MAPS[source])
    outcomes = merge_outcome_types(outcomes, SOURCE_PRIORITY)

    # Type
    type_source_columns = get_source_columns(outcomes, SOURCE_PRIORITY, "type")
    type_comparison, type_disagreements = check_column_agreement(
        outcomes, "type", SOURCE_PRIORITY, fallback_columns=["gtr_gtr_outcome_type"])
    save_disagreements(type_disagreements, type_comparison, "type", "outcomes")
    outcomes.drop(columns=type_source_columns, inplace=True, errors="ignore")
    print(f"{'Merge Rule':<35}: "
          "Majority vote among useful types; "
          "source priority used as fallback; "
          "('other', 'other') used only when no useful type exists")

    # Subtype
    subtype_source_columns = get_source_columns(outcomes, SOURCE_PRIORITY, "subtype")
    subtype_comparison, subtype_disagreements = check_column_agreement(
        outcomes, "subtype", SOURCE_PRIORITY, fallback_columns=["gtr_gtr_outcome_type"])
    save_disagreements(subtype_disagreements, subtype_comparison, 
                       "subtype", "outcomes")
    outcomes.drop(columns=subtype_source_columns, inplace=True, errors="ignore")
    print(f"{'Merge Rule':<35}: "
          "Majority vote among useful types; "
          "source priority used as fallback; "
          "('other', 'other') used only when no useful type exists")

    # ABSTRACT, DESCRIPTION AND TITLES MERGE
    outcomes = merge_title_and_abstract(outcomes)

    # YEAR MERGE
    year_comparison, year_disagreements = check_column_agreement(
        outcomes, "year", SOURCE_PRIORITY)
    year_priority = get_source_priority_by_outliers(outcomes, "year", SOURCE_PRIORITY)
    source_columns = get_source_columns(outcomes, year_priority, "year")
    outcomes = merge_majority_column(outcomes, "year", year_priority)
    print(f"{'Merge Rule':<35}: First populated source "
          f"({' → '.join(source.title() for source in year_priority)})")
    save_disagreements(year_disagreements, year_comparison, "year", "outcomes")
    outcomes.drop(columns=source_columns, inplace=True, errors="ignore")

    # PUBLICATION DATE MERGE
    date_comparison, date_disagreements = check_column_agreement(
        outcomes, "publication_date", SOURCE_PRIORITY)
    date_priority = get_source_priority_by_outliers(outcomes, 
                                                    "publication_date", SOURCE_PRIORITY)
    source_columns = get_source_columns(outcomes, date_priority, "publication_date")
    outcomes = merge_majority_column(outcomes, "publication_date", date_priority)
    print(f"{'Merge Rule':<35}: First populated source "
          f"({' → '.join(source.title() for source in date_priority)})")
    save_disagreements(date_disagreements, date_comparison, "year", "outcomes")
    outcomes.drop(columns=source_columns, inplace=True, errors="ignore")

    # DOI MERGE
    doi_comparison, doi_disagreements = check_column_agreement(
        outcomes, "doi", sources)
    source_columns = get_source_columns(outcomes, SOURCE_PRIORITY, "doi")
    outcomes = merge_preferred_column(outcomes, "doi", source_columns)
    print(f"{'Merge Rule':<35}: First populated source (GtR → Scopus → WoS → OpenAlex)")
    save_disagreements(doi_disagreements, doi_comparison, "doi", "outcomes")
    outcomes.drop(columns=source_columns, inplace=True, errors="ignore")

    # ORGANISATION MERGE
    outcomes, org_disagreements = merge_organisations_by_priority(outcomes)

    # COMPARE FINAL SOURCE TITLE WITH FINAL JOURNAL
    outcomes = merge_source_title_and_journal(outcomes)

    # ISSN MERGE
    outcomes = fill_wos_issns_from_scopus(outcomes)

    # REFERENCE COUNT MERGE
    outcomes = merge_count(outcomes, "reference_count", ["scopus", "wos"])

    # CITED BY MERGE
    outcomes = merge_count(outcomes, "cited_by", ["scopus", "openalex"])

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
    print_summary_header("URL Summary:")
    print(f"{'Empty Source URLs':<35}: {empty_url_count:,}")
    print(f"{'Invalid Source URLs':<35}: {invalid_url_count:,}")
    print(f"{'URL Disagreements':<35}: {len(url_disagreements):,}")
    print(f"{'DOI Fallbacks':<35}: {doi_fallback_count:,}")
    print(f"{'No Final URL':<35}: {no_url_count:,}")
    print(f"{'Merge Rule':<35}: First valid URL by source priority (GtR → Scopus → WoS → OpenAlex); DOI used as fallback")

    # Save url disagreements
    if url_disagreements:
        disagreement_file = OUTCOME_DISAGREEMENT_DIR / "url_disagreements.csv"
        pd.DataFrame(url_disagreements).to_csv(
            disagreement_file, index=False, encoding="utf-8")
        print(f"{'Disagreements Saved':<35}: {disagreement_file.name}")

    # Remove source url columns
    source_url_columns = [
        column
        for column in outcomes.columns
        if column.endswith("_url")
        and column != "url"
    ]

    outcomes.drop(
        columns=source_url_columns,
        inplace=True,
        errors="ignore"
    )

    # KEYWORD COVERAGE
    keyword_summary(outcomes)

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
        "funding_grant_ids",
        "gtr_outcome_type"
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
    return outcomes


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
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