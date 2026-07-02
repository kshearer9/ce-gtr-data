import pandas as pd
import html
import re

# ==========================================
# Files
# ==========================================

INPUT_FILE = "gtr_ce_projects_enriched.xlsx"
OUTPUT_FILE = "gtr_ce_projects_clean.xlsx"

# ==========================================
# Read
# ==========================================

df = pd.read_excel(INPUT_FILE)

# ==========================================
# Encoding / HTML fixes
# ==========================================

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

modified_cells = 0


def fix_text(value):
    global modified_cells

    if pd.isna(value):
        return value

    text = str(value)
    original = text

    text = html.unescape(text)

    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)

    text = " ".join(text.split())

    if text != original:
        modified_cells += 1

    return text


for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(fix_text)

# ==========================================
# Missing Values
# ==========================================

print("\nMissing Values")
print("-" * 40)

missing = df.isna().sum()

for col, value in missing.items():
    if value > 0:
        print(f"{col:<30}{value}")

# ==========================================
# Duplicate Checks
# ==========================================

print("\nDuplicate Checks")
print("-" * 40)

if "project_id" in df.columns:
    print("Duplicate project_id :", df["project_id"].duplicated().sum())

if "title" in df.columns:
    print("Duplicate title :", df["title"].duplicated().sum())

# ==========================================
# Create title_clean
# ==========================================

def normalise_title(text):

    if pd.isna(text):
        return text

    text = text.lower()

    text = re.sub(r"[–—-]", " ", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


if "title" in df.columns:
    df["title_clean"] = df["title"].apply(normalise_title)

# ==========================================
# Save
# ==========================================

df.to_excel(OUTPUT_FILE, index=False)

print("\n" + "=" * 40)
print("GTR data cleaning completed.")
print(f"Modified cells : {modified_cells}")
print(f"Rows           : {len(df)}")
print(f"Columns        : {len(df.columns)}")
print(f"Saved          : {OUTPUT_FILE}")
print("=" * 40)