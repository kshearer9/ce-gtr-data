import pandas as pd
import html
import re
import unicodedata
import numpy as np
from nameparser import HumanName
from utils.constants import REPLACEMENTS

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
    # Convert common bullet symbols to "-"
    text = re.sub(r"[•●▪◦]", "-", text)
    # Remove decorative formatting
    text = re.sub(r"%{3,}", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Convert empty, punctuation-only, or symbol-only values to missing
    if text == "" or all(
        unicodedata.category(char)[0] in {"P", "S"} or char.isspace()
        for char in text):
        return np.nan
    return text