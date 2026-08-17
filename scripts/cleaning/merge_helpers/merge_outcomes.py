import pandas as pd
from utils.merge_config import (
    OUTCOME_DISAGREEMENT_DIR,
    SOURCE_PRIORITY,
    TYPE_MAPS
)
from utils.merge_type_map import GTR_TYPE_MAP
from scripts.cleaning.merge_helpers.merge_helpers import (
    print_summary_header,
    clean_text_value,
    get_source_columns,
    merge_preferred_column,
    merge_longest_column,
    majority_or_priority,
    merge_majority_column,
    get_source_priority_by_outliers,
    get_valid_type_subtype_pairs
)
from scripts.cleaning.merge_helpers.comparison import (
    normalise_for_comparison,
    check_column_agreement,
    save_disagreements
)
from scripts.cleaning.merge_helpers.url_utils import merge_urls

# ---------------------------------------------------------------------------
# OUTCOME COLUMN MERGING HELPERS
# ---------------------------------------------------------------------------

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
    print(f"{'Sources Combined':<35}: {len(available_columns):,}")
    print(f"{'Records with Organisations':<35}: {records_with_organisations:,}")
    print(f"{'Records with Disagreements':<35}: {len(disagreements_df):,}")
    print(f"{'Merge Rule':<35}: First populated source (GtR → Scopus → WoS → OpenAlex)")
    df.drop(columns=available_columns, inplace=True, errors="ignore")
    return df, disagreements_df

def merge_issn_and_eissn(df):
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
        disagreement_file = OUTCOME_DISAGREEMENT_DIR / "issn_disagreements.csv"
        disagreements_df.to_csv(
            disagreement_file, index=False, encoding="utf-8")
        print(f"{'Disagreements Saved':<35}: "
              f"{disagreement_file.name}")
        
    df.drop(columns=["scopus_issn"], inplace=True, errors="ignore")
    df.rename(columns={"wos_issn": "issn", "wos_eissn": "eissn"},
              inplace=True)
    return df


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
        columns=source_title_columns + journal_columns + ["journal"],
        inplace=True, errors="ignore")
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
    Merge type/subtype pairs using majority vote, with source priority
    as the tie-breaker.
    """
    if sources is None:
        sources = SOURCE_PRIORITY
    valid_pairs = get_valid_type_subtype_pairs(TYPE_MAPS)
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
            source_pair = (source_type, source_subtype)
            if source_pair not in valid_pairs:
                continue
            if source_pair == ("other", "other"):
                continue
            values[source] = source_pair
        if not values:
            final_types.append("other")
            final_subtypes.append("other")
            continue

        selected_value = majority_or_priority(values, sources)
        selected_type, selected_subtype = selected_value
        final_types.append(selected_type)
        final_subtypes.append(selected_subtype)
    df["type"] = final_types
    df["subtype"] = final_subtypes

    # GtR fallback for records with no useful type/subtype.
    if "gtr_gtr_outcome_type" in df.columns:
        for index, row in df.iterrows():
            current_type = row["type"]
            current_subtype = row["subtype"]
            type_missing = (
                pd.isna(current_type)
                or not str(current_type).strip()
                or str(current_type).strip().lower() == "other")
            subtype_missing = (
                pd.isna(current_subtype)
                or not str(current_subtype).strip()
                or str(current_subtype).strip().lower() == "other")
            if not (type_missing and subtype_missing):
                continue
            gtr_value = row["gtr_gtr_outcome_type"]
            if pd.isna(gtr_value) or not str(gtr_value).strip():
                continue
            gtr_value = str(gtr_value).strip().lower()
            if gtr_value not in GTR_TYPE_MAP:
                continue
            mapped_type, mapped_subtype = GTR_TYPE_MAP[gtr_value]
            if (mapped_type, mapped_subtype) in valid_pairs:
                df.at[index, "type"] = mapped_type
                df.at[index, "subtype"] = mapped_subtype
    return df

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

        selected_value = majority_or_priority(values, sources)

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

def merge_count(df, column, sources, comparison_sources=None):
    """
    Merge a count column using source priority and report source comparisons.
    `sources` controls which sources are used to create the final merged
    column.
    `comparison_sources` controls which sources are included in the
    comparison statistics. 
    """
    # columns used to create final merged column
    source_columns = get_source_columns(df, sources, column)
    if not source_columns:
        print(f"No source columns found for {column}.")
        return df

    # columns included in the comparison statistics
    if comparison_sources is None:
        comparison_sources = sources
    comparison_columns = get_source_columns(
        df, comparison_sources, column)
    comparison, disagreements = check_column_agreement(
        df, column, sources)

    # Report coverage for every source included in the comparison.
    for source in comparison_sources:
        source_column = f"{source}_{column}"
        if source_column not in df.columns:
            continue
        present = (
            df[source_column].notna()
            & df[source_column].astype(str).str.strip().ne(""))
        print(f"{source.title() + ' counts':<35}: {present.sum():,}")

    # Compare every pair of sources included in the comparison.
    for i, first in enumerate(comparison_columns):
        for second in comparison_columns[i + 1:]:
            first_source = first.removesuffix(f"_{column}")
            second_source = second.removesuffix(f"_{column}")
            both_present = (df[first].notna() & df[second].notna())
            if not both_present.any():
                continue
            difference = df[first] - df[second]
            absolute_difference = difference.abs()
            first_higher = (both_present & (difference > 0))
            second_higher = (both_present & (difference < 0))
            exact_match = (both_present & (difference == 0))
            less_than_3 = (both_present & (absolute_difference > 0)
                           & (absolute_difference < 3))
            three_or_more = (both_present & (absolute_difference >= 3))
            median_difference = difference[both_present].median()
            median_absolute_difference = (
                absolute_difference[both_present].median())
            print()
            print(f"{first_source.title()} vs "
                  f"{second_source.title()}:")
            print(f"{'Both sources present':<35}: "
                  f"{both_present.sum():,}")
            print(f"{first_source.title() + ' higher':<35}: "
                  f"{first_higher.sum():,} "
                  f"({first_higher.sum() / both_present.sum() * 100:.1f}%)")
            print(f"{second_source.title() + ' higher':<35}: "
                  f"{second_higher.sum():,} "
                  f"({second_higher.sum() / both_present.sum() * 100:.1f}%)")
            print(f"{'Exact match':<35}: "
                  f"{exact_match.sum():,} "
                  f"({exact_match.sum() / both_present.sum() * 100:.1f}%)")
            print(f"{'Difference 1–2':<35}: "
                  f"{less_than_3.sum():,} "
                  f"({less_than_3.sum() / both_present.sum() * 100:.1f}%)")
            print(f"{'Difference 3+':<35}: "
                  f"{three_or_more.sum():,} "
                  f"({three_or_more.sum() / both_present.sum() * 100:.1f}%)")
            print(f"{'Median difference':<35}: "
                  f"{median_difference:.1f}")
            print(f"{'Median absolute difference':<35}: "
                  f"{median_absolute_difference:.1f}")

    # Report how often all available sources agree.
    if len(comparison_columns) >= 2:
        available_all = df[comparison_columns].notna().all(axis=1)
        all_agree = (df.loc[available_all, comparison_columns]
                     .nunique(axis=1).eq(1))
        print()
        print(f"{'All comparison sources present':<35}: "
              f"{available_all.sum():,}")
        print(f"{'All comparison sources agree':<35}: "
              f"{all_agree.sum():,}")

    # Merge the final column using only the requested merge sources.
    df = merge_preferred_column(df, column, source_columns)
    print(f"{'Merge Rule':<35}: First populated source "
          f"({' → '.join(source.title() for source in sources)})")
    excluded_from_merge = [source for source in comparison_sources
                           if source not in sources]
    if excluded_from_merge:
        print(f"{'Comparison-only sources':<35}: "
              f"{', '.join(source.title() for source in excluded_from_merge)}")
    save_disagreements(disagreements, comparison, column, "outcomes")
    df.drop(columns=source_columns, inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# OUTCOME MERGE
# ---------------------------------------------------------------------------

def attach_outcome_sources(outcome_map, source_data):
    outcomes = outcome_map.copy()
    outcomes.drop(
        columns=["source", "match_basis", "project_id"],
        inplace=True, errors="ignore")
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
            outcomes[source_id_column].astype("string").str.strip())
        source_df["outcome_id"] = (
            source_df["outcome_id"].astype("string").str.strip())
        # Keep only unique outcomes
        source_df = source_df.drop_duplicates(
            subset=["outcome_id"], keep="first")
        source_df = source_df.rename(
            columns={
                col: f"{source}_{col}"
                for col in source_df.columns
                if col != "outcome_id"})
        source_df = source_df.rename(
            columns={"outcome_id": source_id_column})
        outcomes = outcomes.merge(
            source_df, on=source_id_column, how="left")
        source_id_columns.append(source_id_column)
        sources.append(source)
    # Remove source-specific ID columns
    outcomes.drop(
        columns=source_id_columns, inplace=True, errors="ignore")
    return outcomes, sources

def merge_outcomes(gtr_outcomes, openalex_outcomes, scopus_outcomes,
                             wos_outcomes, outcome_map, validation_lookup):
    """
    Build the global outcome dataset using the existing outcome map.
    """
    source_data = [
            ("gtr", gtr_outcomes),
            ("openalex", openalex_outcomes),
            ("scopus", scopus_outcomes),
            ("wos", wos_outcomes)
        ]
    outcomes, sources = attach_outcome_sources(outcome_map, source_data)

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

    # ABSTRACT, DESCRIPTION AND TITLES MERGE
    outcomes = merge_title_and_abstract(outcomes)

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
    save_disagreements(date_disagreements, date_comparison, "publication_date", "outcomes")
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
    outcomes = merge_issn_and_eissn(outcomes)

    # REFERENCE COUNT MERGE
    outcomes = merge_count(outcomes, "reference_count", ["wos", "scopus"])

    # CITED BY MERGE
    outcomes = merge_count(outcomes, "cited_by", ["wos", "scopus"], ["scopus", "wos", "openalex"])

    # URL MERGE
    outcomes = merge_urls(outcomes, validation_lookup)

    # KEYWORD COVERAGE
    keyword_summary(outcomes)

    # FINAL CLEANUP
    # Remove source prefix from selected final fields
    columns_to_unprefix = [
        "impact",
        "start_year",
        "end_year",
        "eid",
        "pubmed_id",
        "sdg_categories",
        "issue",
        "volume",
        "start_page",
        "end_page",
    ]
    rename_columns = {}
    for source in SOURCE_PRIORITY:
        for column in columns_to_unprefix:
            source_column = f"{source}_{column}"
            if source_column in outcomes.columns:
                rename_columns[source_column] = column
    outcomes.rename(columns=rename_columns, inplace=True)

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
        "gtr_outcome_type",
        "times_cited_core"
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