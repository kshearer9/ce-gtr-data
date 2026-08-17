import pandas as pd
from scripts.cleaning.merge_helpers import validate_urls
from utils.merge_config import (
    OUTCOME_DISAGREEMENT_DIR,
    SOURCE_PRIORITY,
    PROC_DIR
)
from scripts.cleaning.merge_helpers.merge_helpers import print_summary_header

URL_VALIDATION_FILE = PROC_DIR / "unique_url_validation.csv"

# ---------------------------------------------------------------------------
# URL VALIDATION
# ---------------------------------------------------------------------------

def ensure_url_validation_exists(refresh=False):
    """Run URL validation if required."""
    if URL_VALIDATION_FILE.exists() and not refresh:
        print()
        print("Using existing URL validation cache:")
        print(URL_VALIDATION_FILE)
        return
    elif not refresh:
        print()
        print("-" * 70)
        print("URL VALIDATION CACHE NOT FOUND")
        print("-" * 70)
    if URL_VALIDATION_FILE.exists() and not URL_VALIDATION_FILE.is_file():
        raise RuntimeError(
            f"URL validation path exists but is not a file:\n"
            f"{URL_VALIDATION_FILE}")
    print()
    validate_urls.main(refresh=refresh)
    if not URL_VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "validate_urls.py completed but did not create "
            "unique_url_validation.csv.")

def load_url_validation():
    """Load unique_url_validation.csv into a dictionary keyed by URL."""
    if not URL_VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "URL validation file does not exist:\n"
            f"{URL_VALIDATION_FILE}")
    validation_df = pd.read_csv(
        URL_VALIDATION_FILE, encoding="utf-8", dtype=str)
    required_columns = {"url_original", "classification"}
    missing_columns = required_columns - set(validation_df.columns)
    if missing_columns:
        raise ValueError(
            "unique_url_validation.csv is missing:\n"
            + "\n".join(f"  - {column}"
                        for column in sorted(missing_columns)))
    validation_lookup = {}
    for _, row in validation_df.iterrows():
        url = str(row["url_original"]).strip()
        if not url:
            continue
        validation_lookup[url] = {
            "classification": str(row["classification"]).strip().lower(),
            "reason": str(row.get("reason", ""))}
    return validation_lookup


def is_usable_validation(url, validation_lookup):
    """Return True when the URL is classified as valid."""
    if not url:
        return False
    result = validation_lookup.get(url)
    if result is None:
        return False
    return result["classification"] == "valid"


def make_doi_url(doi):
    """Convert a DOI identifier into an HTTPS DOI URL."""
    if pd.isna(doi):
        return ""
    doi = str(doi).strip()
    if not doi:
        return ""
    return f"https://doi.org/{doi}"


def select_final_url(source_urls, doi, validation_lookup=None):
    """Select a URL according to the requested validation mode."""
    for source in SOURCE_PRIORITY:
        url = source_urls.get(source, "")
        if not url:
            continue
        # No validation requested:
        # take the first available URL.
        if validation_lookup is None:
            return url, source
        # Validation requested:
        # take the first URL classified as valid.
        if is_usable_validation(url, validation_lookup):
            return url, source
    # DOI fallback
    doi_url = make_doi_url(doi)
    if doi_url:
        return doi_url, "doi"
    return "", ""


def merge_urls(df, validation_lookup=None):
    final_urls = []
    url_disagreements = []

    invalid_url_count = 0
    empty_url_count = 0
    doi_fallback_count = 0
    no_url_count = 0

    for _, row in df.iterrows():
        source_urls = {}

        # Collect source URLs
        for source in SOURCE_PRIORITY:
            column = f"{source}_url"

            if column not in df.columns:
                source_urls[source] = ""
                continue

            value = row[column]
            source_urls[source] = (
                "" if pd.isna(value) else str(value).strip()
            )

        # Count empty and invalid source URLs
        for url in source_urls.values():
            if not url:
                empty_url_count += 1
            elif validation_lookup is not None:
                if not is_usable_validation(url, validation_lookup):
                    invalid_url_count += 1

        # Check for URL disagreement
        non_empty_urls = {url for url in source_urls.values() if url}

        if len(non_empty_urls) > 1:
            url_disagreements.append({
                "global_outcome_id": row["global_outcome_id"],
                **{
                    f"{source}_url": url
                    for source, url in source_urls.items()
                    if url
                }})

        # Select final URL
        final_url, url_source = select_final_url(
            source_urls, row["doi"], validation_lookup)
        final_urls.append(final_url)
        if url_source == "doi":
            doi_fallback_count += 1
        elif not final_url:
            no_url_count += 1
    df["url"] = final_urls

    # Count how many rows have at least one URL (for reporting)
    rows_with_url = 0
    for _, row in df.iterrows():
        source_urls = {}
        for source in SOURCE_PRIORITY:
            column = f"{source}_url"
            if column not in df.columns:
                source_urls[source] = ""
                continue
            value = row[column]
            source_urls[source] = (
                "" if pd.isna(value) else str(value).strip())
        if any(source_urls.values()):
            rows_with_url += 1

    # Report url statistics
    print_summary_header("URL Summary:")
    print(f"{'Rows with at least one URL':<35}: {rows_with_url:,}")
    if validation_lookup:
        print(f"{'Invalid Source URLs':<35}: {invalid_url_count:,}")
    print(f"{'URL Disagreements':<35}: {len(url_disagreements):,}")
    print(f"{'DOI Fallbacks':<35}: {doi_fallback_count:,}")
    print(f"{'No Final URL':<35}: {no_url_count:,}")
    if validation_lookup is None:
        merge_rule = (
            "First available URL by source priority "
            "(GtR → Scopus → WoS → OpenAlex); DOI used as fallback")
    else:
        merge_rule = (
            "First valid URL by source priority "
            "(GtR → Scopus → WoS → OpenAlex); DOI used as fallback")

    print(f"{'Merge Rule':<35}: {merge_rule}")
    

    # Save url disagreements
    if url_disagreements:
        disagreement_file = OUTCOME_DISAGREEMENT_DIR / "url_disagreements.csv"
        pd.DataFrame(url_disagreements).to_csv(
            disagreement_file, index=False, encoding="utf-8")
        print(f"{'Disagreements Saved':<35}: {disagreement_file.name}")

    # Remove source url columns
    source_url_columns = [
        column
        for column in df.columns
        if column.endswith("_url")
        and column != "url"
    ]

    df.drop(columns=source_url_columns, inplace=True, errors="ignore")
    return df