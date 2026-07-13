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

def csv_to_df(outcome_type):
    input_file = INPUT_DIR / f"gtr_{outcome_type}_latest.csv"
    return pd.read_csv(input_file)

def convert_timestamp(df, columns):
    """
    Convert Unix timestamps in milliseconds to pandas datetime format.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                unit="ms",
                errors="coerce"
            )
    return df


def clean_text(value):
    """
    Clean scraped text while preserving meaningful structure such as
    headings and bullet points.
    """
    if pd.isna(value):
        return np.nan
    text = str(value)
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
# FUNCTIONS TO CLEAN EACH OUTCOME TYPE
# ---------------------------------------------------------------------------

def artisticandcreativeproducts(df):
    # Drop duplicate outcomes
    df = df.drop_duplicates("id")
    # Drop unnecessary columns
    df = df.drop(columns=["href", "outcome_type", "ext", "outcomeid", 
                          "created", "updated", "links.link"],
                          errors="ignore")
    # Rename columns
    df = df.rename(columns={"yearFirstProvided": "year", 
                            "supportingUrl": "supporting_url"})
    # Convert empty strings to NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Fix variable types
    string_cols = ["project_id", "grant_reference", "project_title", "id",
                   "supporting_url"]
    for col in string_cols:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["type"] = df["type"].astype("category")
    # Clean text
    text_columns = ["title", "description", "impact"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[f"{col}_clean"] = (df[col].apply(clean_text).astype("string"))
    return df


def collaborations(df):
    # Drop duplicate outcomes
    df = df.drop_duplicates("id")  
    # Drop unnecessary columns
    df = df.drop(columns=["href", "outcome_type", "ext", "outcomeid", 
                          "created", "updated", "links.link"],
                          errors="ignore")   
    # Rename columns
    df = df.rename(columns={"parentOrganisation": "parent_org", 
                            "childOrganisation": "child_org",
                            "principalInvestigatorContribution": "pi_contribution",
                            "partnerContribution": "partner_contribution",
                            "supportingUrl": "supporting_url"})
    # Convert empty strings to NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Fix variable types
    string_cols = ["project_id", "grant_reference", "project_title", "id",
                   "parent_org", "child_org", "supporting_url"]
    for col in string_cols:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    date_columns = ["start", "end"]
    df = convert_timestamp(df, date_columns)
    category_cols = ["sector", "country"]
    for col in category_cols:
        df[col] = df[col].astype("category")
    # Clean text
    text_columns = ["title", "description", "pi_contribution", 
                    "partner_contribution", "impact"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[f"{col}_clean"] = (df[col].apply(clean_text).astype("string"))
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
        df = cleaning_function(df)
        output_file = OUTPUT_DIR / f"gtr_{outcome_type}_clean.csv"
        df.to_csv(output_file, index = False, encoding = "utf-8")
        print(f"Saved: {output_file}")
    
if __name__ == "__main__":
    main()
