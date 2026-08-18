"""
Merge outcome records from GtR, OpenAlex, Scopus and Web of Science.

Matches records using project ID + DOI, falling back to title or description
when a record has no DOI, combines matched source IDs, and assigns a
global_outcome_id to each unique outcome. Outcomes linked to multiple
projects are retained as separate project rows.
"""

from pathlib import Path
import pandas as pd
import re
from utils.col_types import OUTCOME_COLUMN_TYPES, read_csv

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "cleaned"

INPUT_DIR = DATA_DIR / "outcomes"
OUTPUT_DIR = DATA_DIR / "merged"

for directory in (INPUT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def normalise_text(value):
    """
    Normalise text for matching.
    """
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def normalise_identifier(value):
    """
    Normalise project IDs for matching.
    """
    if pd.isna(value):
        return ""
    value = str(value).lower()
    return re.sub(r"[^a-z0-9]", "", value)

def normalise_doi(value):
    """
    Lower-case a DOI and strip any resolver prefix, for matching.
    """
    if pd.isna(value):
        return ""
    value = re.sub(
        r"^https?://(dx\.)?doi\.org/", "", str(value).strip().lower())
    return value

def create_global_outcome_ids(df):
    """
    Assign one global_outcome_id to each unique outcome.

    The same source outcome ID appearing across multiple project rows
    receives the same global_outcome_id.

    Source IDs are namespaced, so identical IDs from different sources
    cannot accidentally be treated as the same outcome.
    """
    df = df.copy()

    source_id_columns = [
        "gtr_outcome_id",
        "openalex_outcome_id",
        "scopus_outcome_id",
        "wos_outcome_id"
    ]

    source_id_columns = [
        col for col in source_id_columns
        if col in df.columns
    ]

    for col in source_id_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    parent = {}
    next_id = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            parent[root_y] = root_x

    for _, row in df.iterrows():
        source_ids = []

        for col in source_id_columns:
            value = row[col]

            if pd.notna(value):
                source = col.replace("_outcome_id", "")
                source_id = (source, str(value))
                source_ids.append(source_id)

                if source_id not in parent:
                    parent[source_id] = source_id

        if len(source_ids) > 1:
            first_id = source_ids[0]

            for source_id in source_ids[1:]:
                union(first_id, source_id)

    root_to_global_id = {}

    for source_id in parent:
        root = find(source_id)

        if root not in root_to_global_id:
            root_to_global_id[root] = f"OUT{next_id:06d}"
            next_id += 1

    global_ids = []

    for _, row in df.iterrows():
        source_ids = []

        for col in source_id_columns:
            value = row[col]

            if pd.notna(value):
                source = col.replace("_outcome_id", "")
                source_ids.append((source, str(value)))

        if not source_ids:
            global_ids.append(pd.NA)
            continue

        root = find(source_ids[0])
        global_ids.append(root_to_global_id[root])

    df.insert(0, "global_outcome_id", global_ids)

    return df

def prepare_dataframe(df, description_column, source):
    """
    Prepare a dataset for matching.
    """
    df = df.copy()

    # Outcome ID
    if "outcome_id" in df.columns:
        df[f"{source}_outcome_id"] = df["outcome_id"]

    # Project ID
    df["project_id_clean"] = (
        df["project_id"].apply(normalise_identifier)
    )

    # Title
    df["title_clean_for_match"] = (
        df["title_clean"].apply(normalise_text)
    )

    # Description
    df["description_for_match"] = (
        df[description_column].apply(normalise_text)
    )

    # Keep a common description column
    df["description_clean"] = df[description_column]

    # DOI
    df["doi_for_match"] = (
        df["doi"].apply(normalise_doi) if "doi" in df.columns
        else ""
    )

    return df


def create_match_keys(df):
    """
    Create DOI, title and description matching keys.
    """
    df = df.copy()

    # DOI key
    df["project_doi_key"] = (
        df["project_id_clean"]
        + "||"
        + df["doi_for_match"])

    # Title key
    df["project_title_key"] = (
        df["project_id_clean"]
        + "||"
        + df["title_clean_for_match"])

    # Description key
    df["project_description_key"] = (
        df["project_id_clean"]
        + "||"
        + df["description_for_match"])
    return df


def add_gtr_metadata(gtr_df):
    """
    Add source and match type to GTR records.
    Every GTR outcome is retained.
    match_basis means which field is available
    for matching:
        title
        description
        impact
        other
    """
    gtr_df = gtr_df.copy()
    gtr_df["source"] = "gtr"
    gtr_df["match_basis"] = ""

    # Title
    has_title = (gtr_df["title_clean_for_match"] != "")

    gtr_df.loc[has_title, "match_basis"] = "title"

    # Description fallback
    has_description = (gtr_df["description_for_match"] != "")
    gtr_df.loc[
        ~has_title & has_description,
        "match_basis"
    ] = "description"

    # Everything else
    gtr_df.loc[
        ~has_title & ~has_description,
        "match_basis"
    ] = "other"

    return gtr_df


def find_external_matches(
    existing_df,
    external_df,
    external_source
):
    """
    Match an external outcome dataset against an existing outcome dataset.

    Matching order:
    1. project ID + DOI
    2. project ID + title
    3. project ID + description

    DOI goes first: it's the standard identifier for the same output
    appearing in two sources, and it catches cases title/description
    matching misses on their own

    If a match is found:
        - add the external outcome ID to the existing row
        - add the external source to the source column

    If no match is found:
        - add the external record as a new row
    """

    existing_df = existing_df.copy()

    doi_lookup = {}
    title_lookup = {}
    description_lookup = {}

    # Create lookup dictionaries from existing outcomes
    for index, row in existing_df.iterrows():
        doi_key = row["project_doi_key"]
        if (row["doi_for_match"] != "" and doi_key not in doi_lookup):
            doi_lookup[doi_key] = index

        title_key = row["project_title_key"]
        if (row["title_clean_for_match"] != "" and title_key not in title_lookup):
            title_lookup[title_key] = index

        description_key = row["project_description_key"]
        if (row["description_for_match"] != ""
            and description_key not in description_lookup):
            description_lookup[description_key] = index

    matched_indices = set()
    new_external_rows = []

    added_doi_keys = set()
    added_title_keys = set()
    added_description_keys = set()

    # Match external records
    for _, row in external_df.iterrows():
        doi_key = row["project_doi_key"]
        title_key = row["project_title_key"]
        description_key = row["project_description_key"]

        has_doi = (row["doi_for_match"] != "")
        has_title = (row["title_clean_for_match"] != "")
        has_description = (row["description_for_match"] != "")

        matched_index = None
        match_basis = None

        # 1. Try DOI
        if has_doi and doi_key in doi_lookup:
            matched_index = doi_lookup[doi_key]
            match_basis = "doi"

        # 2. Try title
        if (matched_index is None and has_title
            and title_key in title_lookup):
            matched_index = title_lookup[title_key]
            match_basis = "title"

        # 3. Try description
        if (matched_index is None and has_description
            and description_key in description_lookup):
            matched_index = description_lookup[description_key]
            match_basis = "description"

        # Existing outcome matched
        if matched_index is not None:
            matched_indices.add(matched_index)

            # Add external outcome ID
            existing_df.loc[matched_index, f"{external_source}_outcome_id"
                            ] = row[f"{external_source}_outcome_id"]

            # Add source
            current_source = existing_df.loc[matched_index, "source"]
            current_sources = str(current_source).split("; ")
            if external_source not in current_sources:
                existing_df.loc[matched_index, "source"
                ] = f"{current_source}; {external_source}"
            continue

        # No existing match
        if not has_doi and not has_title and not has_description:
            continue

        # Prevent duplicate new records: same project + same DOI/title/
        # description arriving twice within one external source (e.g. two
        # editions of the same journal article) should become one new row,
        # not two, or the second overwrites the first's outcome_id with
        # nothing to show for the first.
        if has_doi:
            if doi_key in added_doi_keys:
                continue
            added_doi_keys.add(doi_key)
            match_basis = "doi"

        elif has_title:
            if title_key in added_title_keys:
                continue
            added_title_keys.add(title_key)
            match_basis = "title"

        elif has_description:
            if description_key in added_description_keys:
                continue
            added_description_keys.add(description_key)
            match_basis = "description"

        # Add as a new outcome
        new_row = row.copy()
        new_row["source"] = external_source
        new_row["match_basis"] = match_basis
        new_external_rows.append(new_row)
    return (existing_df, matched_indices, new_external_rows)


def build_final_dataframe(gtr_df, new_openalex_rows):
    """
    Combine all GTR outcomes with new OpenAlex outcomes.
    Every GTR outcome is retained.
    """
    if new_openalex_rows:
        openalex_new = pd.DataFrame(new_openalex_rows)
        final_result = pd.concat(
            [gtr_df, openalex_new],
            ignore_index=True
        )
    else:
        final_result = gtr_df.copy()

    return final_result


def print_match_summary(
    original_counts,
    match_results,
    final_result
):
    """
    Print a summary of the merge.
    """
    print()
    print("=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)

    # Original database counts
    print("\nORIGINAL OUTCOMES")
    print("-" * 70)
    for source, count in original_counts.items():
        print(f"{source:<15}: {count:>8,}")

    # Matching results
    print("\nMATCHING RESULTS")
    print("-" * 70)
    for source, results in match_results.items():
        matched = len(results["matched"])
        added = len(results["added"])
        print(f"{source:<15}: "
              f"{matched:>8,} matched | "
              f"{added:>8,} added")

    # Final count
    print("\nFINAL DATASET")
    print("-" * 70)
    print(f"{'Total outcomes':<18}: {len(final_result):>8,}")
    print(f"{'Unique outcomes':<18}: {final_result['global_outcome_id'].nunique():>8,}")

    # Source combinations
    print("\nSOURCE COVERAGE")
    print("-" * 70)
    if "source" in final_result.columns:
        source_counts = (final_result["source"].fillna("").value_counts())
        for source_combination, count in source_counts.items():
            print(f"{source_combination:<35}: "
                  f"{count:>8,}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    gtr_file = INPUT_DIR / "gtr_all_outcomes_clean.csv"
    openalex_file = INPUT_DIR / "openalex_all_outcomes_clean.csv"
    scopus_file = INPUT_DIR / "scopus_all_outcomes_clean.csv"
    wos_file = INPUT_DIR / "wos_all_outcomes_clean.csv"

    # Read data
    gtr_df = read_csv(gtr_file, OUTCOME_COLUMN_TYPES)
    openalex_df = read_csv(openalex_file, OUTCOME_COLUMN_TYPES)
    scopus_df = read_csv(scopus_file, OUTCOME_COLUMN_TYPES)
    wos_df = read_csv(wos_file, OUTCOME_COLUMN_TYPES)

    original_counts = {
        "gtr": len(gtr_df),
        "openalex": len(openalex_df),
        "scopus": len(scopus_df),
        "wos": len(wos_df),
    }

    # Prepare data
    gtr_df = prepare_dataframe(gtr_df, "description_clean", "gtr")
    openalex_df = prepare_dataframe(openalex_df, "abstract_clean", "openalex")
    scopus_df = prepare_dataframe(scopus_df, "abstract_clean", "scopus")
    wos_df = prepare_dataframe(wos_df, "abstract_clean", "wos")

    # Create match keys
    gtr_df = create_match_keys(gtr_df)
    openalex_df = create_match_keys(openalex_df)
    scopus_df = create_match_keys(scopus_df)
    wos_df = create_match_keys(wos_df)

    # Add GTR metadata
    gtr_df = add_gtr_metadata(gtr_df)

    # Match OpenAlex to GTR
    final_result, matched_openalex_indices, new_openalex_rows = find_external_matches(
        gtr_df, openalex_df, "openalex")

    # Build final dataset
    final_result = build_final_dataframe(final_result, new_openalex_rows)

    # Add Scopus records
    final_result, matched_scopus_indices, new_scopus_rows = (find_external_matches(
        final_result, scopus_df, "scopus"))
    if new_scopus_rows:
        scopus_new = pd.DataFrame(new_scopus_rows)
        final_result = pd.concat([final_result, scopus_new], ignore_index=True)

    # Add WoS records
    final_result, matched_wos_indices, new_wos_rows = (find_external_matches(
        final_result, wos_df, "wos"))
    if new_wos_rows:
        wos_new = pd.DataFrame(new_wos_rows)
        final_result = pd.concat([final_result, wos_new], ignore_index=True)

    # Create global outcome IDs
    final_result = create_global_outcome_ids(final_result)

    match_results = {
        "openalex": {
            "matched": matched_openalex_indices,
            "added": new_openalex_rows,
        },
        "scopus": {
            "matched": matched_scopus_indices,
            "added": new_scopus_rows,
        },
        "wos": {
            "matched": matched_wos_indices,
            "added": new_wos_rows,
        },
    }

    print_match_summary(
        original_counts,
        match_results,
        final_result
    )

    # Output columns
    output_columns = [
        "project_id",
        "global_outcome_id",
        "gtr_outcome_id",
        "openalex_outcome_id",
        "wos_outcome_id",
        "scopus_outcome_id",
        "match_basis",
        "source"
    ]

    output_columns = [
        column
        for column in output_columns
        if column in final_result.columns
    ]

    final_result = final_result[output_columns]

    output_file = OUTPUT_DIR / "project_outcome_map.csv"
    final_result.to_csv(output_file, index=False, encoding = "utf-8")

    print()
    print(f"Saved project-outcome map to: {output_file}")

    # Final checks
    # Treat blank strings and whitespace as missing
    for column in [
        "project_id",
        "gtr_outcome_id",
        "openalex_outcome_id",
        "scopus_outcome_id",
        "wos_outcome_id",
    ]:
        if column in final_result.columns:
            final_result[column] = (
                final_result[column]
                .replace(r"^\s*$", pd.NA, regex=True)
            )

    # Check for missing project IDs
    missing_project = final_result["project_id"].isna()

    # Check for rows where ALL outcome IDs are missing
    outcome_columns = [
        "gtr_outcome_id",
        "openalex_outcome_id",
        "scopus_outcome_id",
        "wos_outcome_id",
    ]

    missing_all_outcomes = (
        final_result[outcome_columns]
        .isna()
        .all(axis=1)
    )

    # Warnings
    if missing_project.any():
        print(
            f"\nWARNING: {missing_project.sum():,} rows "
            "have no project_id."
        )

    if missing_all_outcomes.any():
        print(
            f"WARNING: {missing_all_outcomes.sum():,} rows "
            "have no outcome ID from any source."
        )

    if not missing_project.any() and not missing_all_outcomes.any():
        print(
            "\nOK: Every row has a project_id and "
            "at least one outcome ID."
        )


if __name__ == "__main__":
    main()