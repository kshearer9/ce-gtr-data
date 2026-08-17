import pandas as pd
import re

# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------

def print_summary_header(title):
    title = title.replace("_", " ").title()
    print()
    print(title)
    print("-" * 70)

def clean_text_value(value):
    """Return a cleaned string, or an empty string for missing values."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""
    return re.sub(r"\s+", " ", value)

def get_source_columns(df, sources, column):
    return [f"{source}_{column}"
            for source in sources
            if f"{source}_{column}" in df.columns]

def get_valid_type_subtype_pairs(type_maps):
    """Return all valid type/subtype pairs from the source mappings."""
    valid_pairs = set()
    for type_map in type_maps.values():
        valid_pairs.update(type_map.values())
    return valid_pairs


# ---------------------------------------------------------------------------
# GENERIC MERGE HELPERS
# ---------------------------------------------------------------------------

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
        if column in df.columns]
    if not available_columns:
        df[output_column] = pd.NA
        return df
    lengths = df[available_columns].map(
        lambda x: len(str(x)) if pd.notna(x) else 0)
    longest_column = lengths.idxmax(axis=1)
    df[output_column] = pd.Series(
        [df.loc[index, column]
        if lengths.loc[index, column] > 0
        else pd.NA
        for index, column in longest_column.items()],
        index=df.index)
    return df

def majority_or_priority(values, sources):
    """
    Select the majority value if one exists.
    Otherwise, use the first available value in source priority order.
    """
    if not values:
        return pd.NA
    value_counts = pd.Series(
        list(values.values()),
        dtype="object"
    ).value_counts()
    majority_values = value_counts[
        value_counts > len(values) / 2
    ]
    if not majority_values.empty:
        return majority_values.index[0]
    for source in sources:
        if source in values:
            return values[source]
    return pd.NA


def merge_majority_column(df, column, sources):
    source_columns = get_source_columns(df, sources, column)
    if not source_columns:
        print(f"No source columns found for {column}.")
        return df
    
    final_values = []
    selected_sources = []
    for index in df.index:
        values = {}
        for source in sources:
            source_column = f"{source}_{column}"
            if source_column not in df.columns:
                continue
            value = df.at[index, source_column]
            if pd.notna(value):
                values[source] = value
        if not values:
            final_values.append(pd.NA)
            selected_sources.append(None)
            continue

        # Count how many sources agree on each value
        value_counts = pd.Series(
            list(values.values()),
            dtype="object"
        ).value_counts()
        majority_values = value_counts[
            value_counts > len(values) / 2]
        if not majority_values.empty:
            selected_value = majority_values.index[0]

            # Find highest-priority source with the selected value
            selected_source = next(
                source
                for source in sources
                if source in values
                and values[source] == selected_value)

        else:
            # No majority: use source priority
            selected_source = next(
                source
                for source in sources
                if source in values)
            selected_value = values[selected_source]

        final_values.append(selected_value)
        selected_sources.append(selected_source)
    df[column] = pd.Series(final_values, index=df.index)

    for source in sources:
        count = sum(
            selected_source == source
            for selected_source in selected_sources)
        print(f"{source.title() + ' selected':<35}: "
              f"{count:,}")
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

