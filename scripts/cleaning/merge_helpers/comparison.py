import re
import unicodedata
import pandas as pd
from utils.merge_config import (
    OUTCOME_DISAGREEMENT_DIR,
    PROJECT_DISAGREEMENT_DIR
)
from scripts.cleaning.merge_helpers.merge_helpers import (
    print_summary_header,
    get_source_columns,
)

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
            if x.strip()}

    return value


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
    print(f"{'Sources Combined':<35}: {len(columns):,}")
    print(f"{f'Records with {column}':<35}: {records_with_column.sum():,}")
    if fallback_columns:
        print(f"{f'Records with {column} after fallback':<35}: "
              f"{records_with_column_or_fallback.sum():,}")
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
