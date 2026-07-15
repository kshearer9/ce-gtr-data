"""
Cleans UKRI Gateway to Research (GtR) outcome datasets.

Pipeline:
- Load processed GtR outcome datasets
- Remove duplicate records and empty values
- Standardise column names and data types
- Clean free-text fields (encoding, HTML, Markdown, URLs, emails)
- Normalise metadata (names, dates, URLs, identifiers)
- Merge equivalent fields into common variables (e.g. year, URL)
- Export cleaned outcome datasets by outcome type and as a combined dataset

Exported Outputs:
- gtr_{outcome_type}_clean.csv - cleaned dataset for each GtR outcome type
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from utils.cleaning import (normalise_name, clean_text, convert_to_numeric,
                            convert_to_category, convert_to_date, convert_to_bool)

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = ROOT_DIR / "data" / "processed" / "gtr" / "outcomes"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CLEANING CONFIGURATION
# ---------------------------------------------------------------------------

COLS_TO_DROP = ["href", "gtr_outcome_type", "ext", "outcomeid", "created", 
                "updated", "links.link"]

RENAME_MAP = {"supportingUrl": "url", "id": "outcome_id"}

STRING_COLS = ["project_id", "grant_reference", "project_title", "outcome_id",
               "supporting_url", "title", "description", "impact", "url"]

TEXT_COLUMNS = ["title", "description", "impact"]

ALL_OUTCOMES_DROP_COLS = ["providedToOthers", "journalTitle", "pubMedId", 
                          "isbn", "issn", "seriesNumber", "seriesTitle", 
                          "subTitle", "volumeTitle", "volumeNumber", "issue", 
                          "totalPages", "edition", "chapterNumber", "chapterTitle", 
                          "pageReference", "conferenceEvent", "conferenceLocation", 
                          "conferenceNumber", "parentOrganisation", 
                          "childOrganisation", "principalInvestigatorContribution", 
                          "partnerContribution", "sector", "country", "form", 
                          "primaryAudience", "results", "typeOfPresentation", 
                          "geographicReach", "partOfOfficialScheme", "narrative", 
                          "organisation", "department", "fundingId", 
                          "amount.currencyCode", "amount.amount", "influence", 
                          "guidelineTitle", "methods", "areas.item", 
                          "softwareDeveloped", "softwareOpenSourced",  
                          "protection", "patentId", "licensed",  
                          "openSourceLicense", "companyName", 
                          "companyDescription", "registrationNumber",
                          "ipExploitated", "jointVenture", "stage", "status",
                          "clinicalTrial", "ukcrnIsctnId"]


# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------


def clean_doi_and_url(df):
    """
    Clean DOI identifiers and create a single URL column while keeping DOI.
    """
    # Extract DOI identifier from DOI URLs or plain DOI strings
    if "doi" in df.columns:
        df["doi"] = (df["doi"].astype("string").str.extract(
            r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", expand=False))

    # Create URL column
    df["url"] = np.nan
    url_cols = ["supportingUrl", "publicationUrl", "website", "patentUrl"]
    for col in url_cols:
        if col in df.columns:
            df["url"] = df["url"].fillna(df[col])

    # Use cleaned DOI to fill missing URLs
    if "doi" in df.columns:
        doi_url = "https://doi.org/" + df["doi"]
        df["url"] = df["url"].fillna(doi_url)

    # Remove source URL columns, but keep DOI
    df = df.drop(columns=url_cols, errors="ignore")
    return df


def merge_date(df):
    """
    Merge all year/date fields into a single 'year' column.
    1. Explicit year fields
    3. Start/end dates (expanded into comma-separated years)
    """
    df["year"] = np.nan
    year_columns = [
        "datePublished",
        "yearFirstProvided",
        "yearsOfDissemination",
        "yearEstablished",
        "yearDevelopmentCompleted",
        "yearProtectionGranted"]
    for col in year_columns:
        if col in df.columns:
            if col == "datePublished":
                df["year"] = df["year"].fillna(df[col].dt.year.astype("string"))
            elif col == "yearsOfDissemination":
                df["year"] = df["year"].fillna(df[col].astype("string").str
                                               .replace(r"\s*,\s*", "; ", regex=True))
            else:
                df["year"] = df["year"].fillna(df[col].astype("string"))
    # If start and end dates provided, convert to same form as years of dissemination
    if "start" in df.columns:
        start_year = df["start"].dt.year
        if "end" in df.columns:
            end_year = df["end"].dt.year
            missing_year = df["year"].isna()
            df.loc[missing_year, "year"] = df.loc[missing_year].apply(
                lambda row: ("; ".join(str(year)
                        for year in range(
                            int(start_year[row.name]),
                            int(end_year[row.name]) + 1))
                    if pd.notna(end_year[row.name])
                    and pd.notna(start_year[row.name])
                    and end_year[row.name] >= start_year[row.name]
                    else str(int(start_year[row.name]))
                    if pd.notna(start_year[row.name])
                    else pd.NA), axis=1)
        else:
            df["year"] = df["year"].fillna(start_year.astype("string"))
    # Remove original date/year columns
    df = df.drop(columns=year_columns + ["start", "end"], errors="ignore")
    return df
            

# ---------------------------------------------------------------------------
# SHARED CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

def clean_df(df):
    # Remove duplicate outcomes
    if "outcome_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates("outcome_id")
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} duplicate outcomes")
    # Convert empty strings to NaN
    df = df.replace({
        r"(?i)^\s*$": np.nan,
        r"(?i)^nil$": np.nan,
        r"(?i)^null$": np.nan,
        r"(?i)^none$": np.nan,
        r"(?i)^n/?a\.?$": np.nan,
        r"(?i)^n\.?a\.?$": np.nan,
    }, regex=True)
    # Remove empty columns
    df = df.replace("[]", np.nan)
    df = df.dropna(axis=1, how="all")
    return df


def drop_columns(df, outcome_type, *extra_cols):
    """ Drops unnecessary columns. """
    drop_cols = COLS_TO_DROP.copy()
    drop_cols.extend(extra_cols)
    if outcome_type != "all_outcomes":
        drop_cols.append("outcome_type")
    return df.drop(columns=drop_cols, errors="ignore")


def rename_columns(df, extra_map=None):
    """ Renames columns. """
    rename_map = RENAME_MAP.copy()
    if extra_map:
        rename_map.update(extra_map)
    return df.rename(columns=rename_map)


def convert_to_string(df, *extra_cols):
    """ Converts columns to string type. """
    string_cols = STRING_COLS.copy()
    string_cols.extend(extra_cols)
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def clean_text_columns(df, *extra_cols):
    """ Cleans all text columns. """
    text_cols = TEXT_COLUMNS.copy()
    text_cols.extend(extra_cols)
    for col in text_cols:
        if col in df.columns:
            df[f"{col}_clean"] = (df[col].astype("string").apply(clean_text).astype("string"))
    return df


def clean_issn(df):
    if "issn" in df.columns:
        df["issn"] = (df["issn"].astype("string")
                      .str.replace("-", "", regex=False).str.strip())
        # Add hyphen after the first 4 characters
        df["issn"] = df["issn"].str.replace(r"^(\d{4})(\d{3}[\dXx])$",
                                            r"\1-\2", regex=True)
    return df


# ---------------------------------------------------------------------------
# OUTCOME-SPECIFIC CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

def artisticandcreativeproducts(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"yearFirstProvided": "year"})
    df = convert_to_numeric(df, "year")
    df = convert_to_category(df, "type")
    df = clean_text_columns(df)
    df = convert_to_string(df)
    return df


def collaborations(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"parentOrganisation": "parent_org", 
                            "childOrganisation": "child_org",
                            "principalInvestigatorContribution": "pi_contribution",
                            "partnerContribution": "partner_contribution",})
    df = convert_to_date(df, "start", "end")
    df = convert_to_category(df, "sector", "country")
    df = clean_text_columns(df)
    df = convert_to_string(df, "parent_org", "child_org", "pi_contribution",
                        "partner_contribution")
    return df


def disseminations(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = merge_date(df)
    df = rename_columns(df, {"primaryAudience": "primary_audience",
                             "typeOfPresentation": "presentation_type",
                             "geographicReach": "geographic_reach",
                             "partOfOfficialScheme": "part_of_official_scheme"})
    df = convert_to_category(df, "form", "primary_audience", "presentation_type", 
                              "geographic_reach")
    df = convert_to_bool(df, "part_of_official_scheme")
    df = clean_text_columns(df)
    df = convert_to_string(df, "results", "year")
    return df


def furtherfundings(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"fundingId": "further_funding_id",
                             "amount.currencyCode": "currency_code",
                             "amount.amount": "amount"})
    df = convert_to_numeric(df, "amount")
    df = convert_to_date(df, "start", "end")
    df = convert_to_category(df, "organisation", "department", "sector", 
                                  "country", "currency_code")
    df = clean_text_columns(df)
    df = convert_to_string(df, "narrative", "organisation", "department", 
                           "further_funding_id")
    return df


def intellectualproperties(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"patentId": "patent_id",
                             "yearProtectionGranted": "year",
                             "patentUrl": "patent_url"})
    df = convert_to_numeric(df, "year")
    df = convert_to_date(df, "start", "end")
    df = convert_to_category(df, "protection", "type")
    df = convert_to_bool(df, "licensed")
    df = clean_text_columns(df)
    df = convert_to_string(df, "patent_id", "patent_url")
    return df


def policyinfluences(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"guidelineTitle": "guideline_title",
                             "geographicReach": "geographic_reach",
                             "patentUrl": "patent_url"})
    df = convert_to_category(df, "type", "geographic_reach")
    df = convert_to_bool(df, "licensed")
    if "area.item" in df.columns:
        df["policy_areas"] = df["area.item"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else x)
    df = clean_text_columns(df, "influence", "guideline_title", "methods")
    df = convert_to_string(df, "influence", "guideline_title", "methods")
    return df


def products(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"clinicalTrial": "clinical_trial",
                             "ukcrnIsctnId": "clinical_trial_id",
                             "yearDevelopmentCompleted": "year"})
    df = convert_to_numeric(df, "year")
    df = convert_to_category(df, "type", "stage", "status")
    df = clean_text_columns(df)
    df = convert_to_string(df, "clinical_trial", "clinical_trial_id")
    return df


def publications(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"abstractText": "abstract",
                             "otherInformation": "other_info",
                             "journalTitle": "journal_title",
                             "datePublished": "date_published",
                             "publicationUrl": "url",
                             "pubMedId": "pubmed_id",
                             "seriesNumber": "series_num",
                             "seriesTitle": "series_title",
                             "subTitle": "sub_title",
                             "volumeTitle": "vol_title",
                             "volumeNumber": "vol_num",
                             "totalPages": "total_pages",
                             "chapterNumber": "chapter_num",
                             "chapterTitle": "chapter_title",
                             "pageReference": "page_ref",
                             "conferenceEvent": "conference",
                             "conferenceLocation": "conference_location",
                             "conferenceNumber": "conference_num"})
    df = clean_doi_and_url(df)
    df = clean_issn(df)
    df["author_clean"] = df["author"].apply(normalise_name)
    df = convert_to_numeric(df, "total_pages")
    df = convert_to_date(df, "date_published")
    df = convert_to_category(df, "type", "journal_title")
    df = clean_text_columns(df, "abstract", "other_info", "series_title",
                            "sub_title", "volume_title", "chapter_title")
    df = convert_to_string(df, "abstract", "other_info", "series_title",
                            "sub_title", "volume_title", "chapter_title", 
                            "pubmed_id", "isbn", "issn", "series_num", 
                            "vol_num", "issue", "edition", "chapter_num", 
                            "page_ref", "conference", "conference_location",
                            "conference_num", "author")
    return df


def researchdatabaseandmodels(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"providedToOthers": "provided_to_others",
                             "yearFirstProvided": "year"})
    df = convert_to_numeric(df, "year")
    df = convert_to_category(df, "type")
    df = convert_to_bool(df, "provided_to_others")
    df = clean_text_columns(df)
    df = convert_to_string(df)
    return df


def researchmaterials(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"softwareDeveloped": "software_developed",
                             "softwareOpenSourced": "software_open_sourced",
                             "providedToOthers": "provided_to_others",
                             "yearFirstProvided": "year"})
    df = convert_to_numeric(df, "year")
    df = convert_to_category(df, "type")
    df = convert_to_bool(df, "provided_to_others", "software_developed",
                              "software_open_sourced")
    df = clean_text_columns(df)
    df = convert_to_string(df)
    return df


def softwareandtechnicalproducts(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"softwareOpenSourced": "software_open_sourced",
                             "openSourceLicense": "open_source_license",
                             "yearFirstProvided": "year"})
    df = convert_to_numeric(df, "year")
    df = convert_to_category(df, "type")
    df = convert_to_bool(df, "software_open_sourced")
    df = clean_text_columns(df)
    df = convert_to_string(df, "open_source_license")    
    return df


def spinouts(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"companyName": "company_name",
                             "companyDescription": "company_description",
                             "website": "url",
                             "registrationNumber": "reg_num",
                             "yearEstablished": "year",
                             "ipExploited": "ip_exploited",
                             "jointVenture": "joint_venture"})
    df = convert_to_numeric(df, "year")
    df = convert_to_bool(df, "ip_exploited", "joint_venture")
    df = clean_text_columns(df, "company_description")
    df = convert_to_string(df, "company_name", "reg_num")
    return df


def all_outcomes(df, outcome_type):
    df = clean_df(df)
    df = clean_doi_and_url(df)
    df = convert_to_date(df, "datePublished", "start", "end")
    df = merge_date(df)
    df["type"] = df["type"].fillna(df["form"])
    df["organisations"] = (df[["parentOrganisation", "childOrganisation"]]
        .astype("string").apply(lambda x: "; ".join(x.dropna()), axis=1))
    df["author_clean"] = df["author"].apply(normalise_name)
    df = drop_columns(df, outcome_type, *ALL_OUTCOMES_DROP_COLS)
    df = rename_columns(df)
    df = convert_to_category(df, "type")
    df = clean_text_columns(df)
    df = convert_to_string(df, "year", "author", "organisations")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def get_outcome_type(file):
    match = re.search(r"gtr_(.*?)_latest\.csv$", file.name)
    if match:
        return match.group(1)
    return None


def main():
    for file in INPUT_DIR.glob("gtr_*_latest.csv"):
        outcome_type = get_outcome_type(file)
        if outcome_type is None:
            continue
        cleaning_function = globals().get(outcome_type)
        if not callable(cleaning_function):
            print(f"No cleaning function found for '{outcome_type}'\n")
            continue
        print(f"Cleaning: '{outcome_type}'")
        df = pd.read_csv(file, encoding ="utf-8")
        try:
            df = cleaning_function(df, outcome_type)
        except Exception as e:
            print(f"Failed cleaning {outcome_type}: {e}")
            continue
        output_file = OUTPUT_DIR / f"gtr_{outcome_type}_clean.csv"
        df.to_csv(output_file, index = False, encoding = "utf-8")

        print("=" * 40)
        print(f"Rows           : {len(df)}")
        print(f"Columns        : {len(df.columns)}")
        print(f"Saved          : {output_file.name}")
        print("=" * 40 + "\n") 
    

if __name__ == "__main__":
    main()