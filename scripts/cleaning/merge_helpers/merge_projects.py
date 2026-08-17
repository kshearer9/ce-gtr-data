import pandas as pd
from scripts.cleaning.merge_helpers.merge_helpers import print_summary_header
from scripts.cleaning.merge_helpers.comparison import normalise_for_comparison

# ---------------------------------------------------------------------------
# PROJECT MERGE
# ---------------------------------------------------------------------------

def normalise_funding_type(value):
    """Normalise funding types for cross-source comparison."""
    if pd.isna(value):
        return pd.NA
    value = str(value).strip().lower()
    funding_type_map = {
        "research grant": "collaborative r&d",
        "feasibility studies": "research grant",
        "eu-funded": "grant"}
    return funding_type_map.get(value, value)


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
