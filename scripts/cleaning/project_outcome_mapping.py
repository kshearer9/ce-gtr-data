from pathlib import Path
import pandas as pd
import re

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "team-code" / "data" / "cleaned"

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

    return df


def create_match_keys(df):
    """
    Create title and description matching keys.
    """
    df = df.copy()

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

    # Impact fallback
    has_impact = (
        gtr_df["impact_clean"].notna()
        & gtr_df["impact_clean"].astype(str).str.strip().ne("")
    )

    gtr_df.loc[
        ~has_title & ~has_description & has_impact,
        "match_basis"
    ] = "impact"

    # Other
    gtr_df.loc[
        ~has_title & ~has_description & ~has_impact,
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
    1. project ID + title
    2. project ID + description

    If a match is found:
        - add the external outcome ID to the existing row
        - add the external source to the source column

    If no match is found:
        - add the external record as a new row
    """

    existing_df = existing_df.copy()

    title_lookup = {}
    description_lookup = {}

    # Create lookup dictionaries from existing outcomes
    for index, row in existing_df.iterrows():
        title_key = row["project_title_key"]
        if (row["title_clean_for_match"] != "" and title_key not in title_lookup):
            title_lookup[title_key] = index

        description_key = row["project_description_key"]
        if (row["description_for_match"] != "" 
            and description_key not in description_lookup):
            description_lookup[description_key] = index

    matched_indices = set()
    new_external_rows = []

    added_title_keys = set()
    added_description_keys = set()

    # Match external records
    for _, row in external_df.iterrows():
        title_key = row["project_title_key"]
        description_key = row["project_description_key"]

        has_title = (row["title_clean_for_match"] != "")
        has_description = (row["description_for_match"] != "")

        matched_index = None
        match_basis = None

        # 1. Try title
        if has_title and title_key in title_lookup:
            matched_index = title_lookup[title_key]
            match_basis = "title"

        # 2. Try description
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
        if not has_title and not has_description:
            continue

        # Prevent duplicate new title records
        if has_title:
            if title_key in added_title_keys:
                continue
            added_title_keys.add(title_key)
            match_basis = "title"

        # Otherwise use description
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
    original_gtr_count,
    original_openalex_count,
    original_scopus_count,
    matched_openalex_indices,
    matched_scopus_indices,
    new_openalex_rows,
    new_scopus_rows,
    final_result
):
    """
    Print a summary of the merge.
    """

    openalex_matches = len(matched_openalex_indices)
    openalex_added = len(new_openalex_rows)

    scopus_matches = len(matched_scopus_indices)
    scopus_added = len(new_scopus_rows)

    final_count = len(final_result)

    print()
    print("MERGE SUMMARY")
    print("=" * 50)
    print(f"Original GTR outcomes        : {original_gtr_count}")
    print(f"Original OpenAlex outcomes   : {original_openalex_count}")
    print(f"Original Scopus outcomes     : {original_scopus_count}")

    print()
    print(f"OpenAlex matched to existing : {openalex_matches}")
    print(f"New OpenAlex outcomes added  : {openalex_added}")

    print()
    print(f"Scopus matched to existing   : {scopus_matches}")
    print(f"New Scopus outcomes added    : {scopus_added}")

    print()
    print(f"Final total outcomes         : {final_count}")

    print()
    print("MATCH TYPE")
    print("=" * 50)
    print(
        f"Title                       : "
        f"{(final_result['match_basis'] == 'title').sum()}"
    )

    print(
        f"Description                 : "
        f"{(final_result['match_basis'] == 'description').sum()}"
    )

    print(
        f"Impact                      : "
        f"{(final_result['match_basis'] == 'impact').sum()}"
    )

    print(
        f"Other                       : "
        f"{(final_result['match_basis'] == 'other').sum()}"
    )

    print()
    print("SOURCE")
    print("=" * 50)
    print(
        f"GTR only                    : "
        f"{(final_result['source'] == 'gtr').sum()}"
    )

    print(
        f"GTR + OpenAlex              : "
        f"{(final_result['source'] == 'gtr; openalex').sum()}"
    )

    print(
        f"GTR + Scopus                : "
        f"{(final_result['source'] == 'gtr; scopus').sum()}"
    )

    print(
        f"GTR + OpenAlex + Scopus     : "
        f"{(final_result['source'] == 'gtr; openalex; scopus').sum()}"
    )

    print(
        f"OpenAlex only               : "
        f"{(final_result['source'] == 'openalex').sum()}"
    )

    print(
        f"OpenAlex + Scopus           : "
        f"{(final_result['source'] == 'openalex; scopus').sum()}"
    )

    print(
        f"Scopus only                 : "
        f"{(final_result['source'] == 'scopus').sum()}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    gtr_file = INPUT_DIR / "gtr_all_outcomes_clean.csv"
    openalex_file = INPUT_DIR / "openalex_all_outcomes_clean.csv"
    scopus_file = INPUT_DIR / "scopus_all_outcomes_clean.csv"

    # Read data
    gtr_df = pd.read_csv(gtr_file, encoding = "utf-8")
    openalex_df = pd.read_csv(openalex_file, encoding = "utf-8")
    scopus_df = pd.read_csv(scopus_file, encoding = "utf-8")

    original_gtr_count = len(gtr_df)
    original_openalex_count = len(openalex_df)
    original_scopus_count = len(scopus_df)

    # Prepare data
    gtr_df = prepare_dataframe(gtr_df, "description_clean", "gtr")
    openalex_df = prepare_dataframe(openalex_df, "abstract", "openalex")
    scopus_df = prepare_dataframe(scopus_df, "abstract", "scopus")

    # Create match keys
    gtr_df = create_match_keys(gtr_df)
    openalex_df = create_match_keys(openalex_df)
    scopus_df = create_match_keys(scopus_df)

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

    # Print summary
    print_match_summary(
        original_gtr_count,
        original_openalex_count,
        original_scopus_count,
        matched_openalex_indices,
        matched_scopus_indices,
        new_openalex_rows,
        new_scopus_rows,
        final_result
    )

    # Output columns
    output_columns = [
        "gtr_outcome_id",
        "openalex_outcome_id",
        "project_id",
        "title_clean",
        "description_clean",
        "impact_clean",
        "type",
        "doi",
        "author_clean",
        "organisations",
        "year",
        "url",
        "match_basis",
        "source"
    ]

    # Only keep columns that actually exist
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


if __name__ == "__main__":
    main()