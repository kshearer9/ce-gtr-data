from pathlib import Path
import pandas as pd
import numpy as np
import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GTR_DIR = SCRIPT_DIR.parent
INPUT_DIR = GTR_DIR / "data" / "processed" / "outcomes"
OUTPUT_DIR = GTR_DIR / "data" / "cleaned" / "outcomes"

for d in (INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLEANING CONFIGURATION
# ---------------------------------------------------------------------------

REPLACEMENTS = {

    # Common mojibake
    "‚Äì": "–",
    "‚Äî": "—",
    "‚Äò": "'",
    "‚Äô": "'",
    "‚Äú": '"',
    "‚Äù": '"',
    "‚Ä¶": "...",
    "‚Ä¢": "•",
    "‚Ñ¢": "™",

    # Alternative mojibake
    "â€“": "–",
    "â€”": "—",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€¦": "...",
    "â€¢": "•",
    "â„¢": "™",
    "Â£": "£",

    # HTML entities
    "&quot;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&#39;": "'",

    # Extra encoding artefacts
    "\xa0": " ",
    "Â": "",
}

COLS_TO_DROP = ["href", "outcome_type", "ext", "outcomeid", "created", 
                "updated", "links.link"]

RENAME_MAP = {"supportingUrl": "url"}

STRING_COLS = ["project_id", "grant_reference", "project_title", "id",
               "supporting_url", "title", "description", "impact"]

TEXT_COLUMNS = ["title", "description", "impact"]


# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------

def csv_to_df(outcome_type):
    input_file = INPUT_DIR / f"gtr_{outcome_type}_latest.csv"
    return pd.read_csv(input_file)

def clean_text(text):
    """
    Clean scraped text while preserving meaningful structure such as
    headings and bullet points.
    """
    if pd.isna(text):
        return np.nan
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove Markdown formatting
    text = text.replace("**", "")
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Convert common bullet symbols to "-"
    text = re.sub(r"[•●▪◦]", "-", text)
    # Fix encoding artefacts
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Normalise whitespace
    text = " ".join(text.split())
    # Convert empty, punctuation-only, or symbol-only values to missing
    if text == "" or all(
        unicodedata.category(char)[0] in {"P", "S"} or char.isspace()
        for char in text):
        return np.nan
    return text


# ---------------------------------------------------------------------------
# SHARED CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

def clean_df(df):
    # Remove duplicate outcomes
    if "id" in df.columns:
        df = df.drop_duplicates("id")
    # Convert empty strings to NaN
    df = df.replace({r"^\s*$": np.nan, r"(?i)^nil$": np.nan}, regex=True)
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

def convert_to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def convert_to_category(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def convert_to_date(df, columns):
    """
    Convert Unix timestamps in milliseconds to pandas datetime format.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df

def convert_to_bool(df, cols):
    """
    Convert columns to boolean type.
    Handles existing booleans and common text/numeric representations.
    """
    true_values = {"true", "t", "yes", "y", "1", "1.0"}
    false_values = {"false", "f", "no", "n", "0", "0.0"}
    for col in cols:
        if col in df.columns:
            # Already boolean - leave as is
            if df[col].dtype == "bool":
                continue
            # Convert strings to lowercase for matching
            values = df[col].astype("string").str.strip().str.lower()
            df[col] = values.map(lambda x: True if x in true_values
                                 else False if x in false_values 
                                 else pd.NA).astype("boolean")
    return df

def clean_text_columns(df, *extra_cols):
    """ Cleans all text columns. """
    text_cols = TEXT_COLUMNS.copy()
    text_cols.extend(extra_cols)
    for col in text_cols:
        if col in df.columns:
            df[f"{col}_clean"] = (df[col].astype("string").apply(clean_text).astype("string"))
    return df

def clean_doi_and_url(df):
    """
    Extract DOI identifier and fill missing publication URLs using DOI links.
    """
    if "doi" in df.columns:
        # Extract DOI from either a full DOI URL or plain DOI
        df["doi"] = (df["doi"].astype("string")
                     .str.extract(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", 
                                  expand=False))
    if "url" in df.columns:
        # Fill missing publication URLs with DOI URLs
        if "doi" in df.columns:
            doi_url = "https://doi.org/" + df["doi"]
            df["url"] = (df["url"].astype("string").fillna(doi_url))
    return df

def clean_issn(df):
    if "issn" in df.columns:
        df["issn"] = (df["issn"].astype("string")
                      .str.replace("-", "", regex=False).str.strip())
        # Add hyphen after the first 4 characters
        df["issn"] = df["issn"].str.replace(r"^(\d{4})(\d{3}[\dXx])$",
                                            r"\1-\2", regex=True)
    return df

def normalise_name(name):
    if pd.isna(name):
        return pd.NA
    name = str(name)
    # Remove punctuation
    name = re.sub(r"[,.]", "", name)
    # Remove extra whitespace
    name = " ".join(name.split())
    # Remove accents for matching
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.strip()


# ---------------------------------------------------------------------------
# OUTCOME-SPECIFIC CLEANING FUNCTIONS
# ---------------------------------------------------------------------------

def artisticandcreativeproducts(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"yearFirstProvided": "year"})
    df = convert_to_numeric(df, ["year"])
    df = convert_to_category(df, ["type"])
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
    df = convert_to_date(df, ["start", "end"])
    df = convert_to_category(df, ["sector", "country"])
    df = clean_text_columns(df)
    df = convert_to_string(df, "parent_org", "child_org", "pi_contribution",
                        "partner_contribution")
    return df

def disseminations(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"primaryAudience": "primary_audience",
                             "yearsOfDissemination": "year",
                             "typeOfPresentation": "presentation_type",
                             "geographicReach": "geographic_reach",
                             "partOfOfficialScheme": "part_of_official_scheme"})
    df = convert_to_category(df, ["form", "primary_audience", "presentation_type", 
                              "geographic_reach"])
    df = convert_to_bool(df, ["part_of_official_scheme"])
    df = clean_text_columns(df)
    df = convert_to_string(df, "results", "year")
    return df

def furtherfundings(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"fundingId": "further_funding_id",
                             "amount.currencyCode": "currency_code",
                             "amount.amount": "amount"})
    df = convert_to_numeric(df, ["amount"])
    df = convert_to_date(df, ["start", "end"])
    df = convert_to_category(df, ["organisation", "department", "sector", 
                                  "country", "currency_code"])
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
    df = convert_to_numeric(df, ["year"])
    df = convert_to_date(df, ["start", "end"])
    df = convert_to_category(df, ["protection", "type"])
    df = convert_to_bool(df, ["licensed"])
    df = clean_text_columns(df)
    df = convert_to_string(df, "patent_id", "patent_url")
    return df

def policyinfluences(df, outcome_type):
    df = clean_df(df)
    df = drop_columns(df, outcome_type)
    df = rename_columns(df, {"guidelineTitle": "guideline_title",
                             "geographicReach": "geographic_reach",
                             "patentUrl": "patent_url"})
    df = convert_to_category(df, ["type", "geographic_reach"])
    df = convert_to_bool(df, ["licensed"])
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
                             "yearDevelopmentCompleted": "year_completed"})
    df = convert_to_numeric(df, ["year_completed"])
    df = convert_to_category(df, ["type", "stage", "status"])
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
    df = convert_to_numeric(df, ["total_pages"])
    df = convert_to_date(df, ["date_published"])
    df = convert_to_category(df, ["type", "journal_title"])
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
    df = convert_to_numeric(df, ["year"])
    df = convert_to_category(df, ["type"])
    df = convert_to_bool(df, ["provided_to_others"])
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
    df = convert_to_numeric(df, ["year"])
    df = convert_to_category(df, ["type"])
    df = convert_to_bool(df, ["provided_to_others", "software_developed",
                              "software_open_source"])
    df = clean_text_columns(df)
    df = convert_to_string(df)
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
            print(f"No cleaning function found for '{outcome_type}'")
            continue
        print(f"Cleaning: '{outcome_type}'")
        df = pd.read_csv(file, encoding ="utf-8")
        df = cleaning_function(df, outcome_type)
        output_file = OUTPUT_DIR / f"gtr_{outcome_type}_clean.csv"
        df.to_csv(output_file, index = False, encoding = "utf-8")
        print(f"Saved: {output_file}")
    
if __name__ == "__main__":
    main()
