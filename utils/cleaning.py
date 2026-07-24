import pandas as pd
import html
import re
import unicodedata
import numpy as np
from nameparser import HumanName
from utils.constants import REPLACEMENTS, TRUE_VALUES, FALSE_VALUES

def normalise_name(name):
    if pd.isna(name):
        return pd.NA
    name = str(name).strip()
    # Remove commas and full stops
    name = re.sub(r"[,.]", "", name)
    # Collapse whitespace
    name = " ".join(name.split())
    parts = name.split()
    # Already looks like surname and initials
    if len(parts) >= 2 and all(len(p) <= 2 for p in parts[1:]):
        return name
    # Only parse obvious full names
    if len(parts) == 2:
        parsed = HumanName(name)
        if parsed.first and parsed.last:
            return f"{parsed.last} {parsed.first[0]}"
    return name

def clean_text(text):
    """
    Clean text fields for NLP by removing formatting, encoding artefacts,
    HTML, URLs and emails.
    """
    if pd.isna(text):
        return np.nan
    # Decode HTML entities
    text = html.unescape(text)
    # Fix encoding artefacts
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove Markdown formatting
    text = re.sub(r"[*_`#]+", "", text)
    # Convert common bullet symbols to "-" - may change this
    text = re.sub(r"[•●▪◦]", "-", text)
    # Remove decorative formatting
    text = re.sub(r"%{3,}", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)
    # Remove Markdown reference links e.g. [text][0]
    text = re.sub(r"\[[^\]]+\]\[\d+\]", "", text)
    # Remove Markdown link definitions e.g. [0]: URL
    text = re.sub(r"\[\d+\]:.*", "", text)
    # Remove standalone markdown brackets
    text = re.sub(r"[\[\]]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Convert empty, punctuation-only, or symbol-only values to missing
    if text == "" or all(
        unicodedata.category(char)[0] in {"P", "S"} or char.isspace()
        for char in text):
        return np.nan
    return text

def convert_to_string(df, *cols):
    """ Converts columns to string type. """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def convert_to_numeric(df, *cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def convert_to_category(df, *cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def convert_to_date(df, *cols):
    """
    Convert Unix timestamps in milliseconds to pandas datetime format.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def convert_to_bool(df, *cols):
    """
    Convert columns to boolean type.
    Handles existing booleans and common text/numeric representations.
    """
    for col in cols:
        if col in df.columns:
            # Already boolean - leave as is
            if df[col].dtype == "bool":
                continue
            # Convert strings to lowercase for matching
            values = df[col].astype("string").str.strip().str.lower()
            df[col] = values.map(lambda x: True if x in TRUE_VALUES
                                 else False if x in FALSE_VALUES 
                                 else pd.NA).astype("boolean")
    return df


def clean_text_columns(df, *text_cols):
    """ Cleans all text columns. """
    clean_cols = []
    for col in text_cols:
        if col in df.columns:
            clean_col = f"{col}_clean"
            df[clean_col] = (df[col].astype("string").apply(clean_text))
            clean_cols.append(clean_col)
    df = convert_to_string(df, *(list(text_cols) + clean_cols))
    return df