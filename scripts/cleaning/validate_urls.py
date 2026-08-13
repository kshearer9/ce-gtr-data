from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# FILE SETUP
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

OUTCOME_INPUT_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"
PROC_DIR = ROOT_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = PROC_DIR / "unique_url_validation.csv"

SOURCE_FILES = {
    "gtr": "gtr_all_outcomes_clean.csv",
    "scopus": "scopus_all_outcomes_clean.csv",
    "wos": "wos_all_outcomes_clean.csv",
    "openalex": "openalex_all_outcomes_clean.csv",
}

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15
MAX_WORKERS = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchDatasetURLValidator/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.5",
}

# ---------------------------------------------------------------------------
# HTTP SESSION
# ---------------------------------------------------------------------------

thread_local = threading.local()

def get_session():
    """Return a requests session for the current thread."""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        thread_local.session = session

    return thread_local.session


# ---------------------------------------------------------------------------
# RESULT CREATION
# ---------------------------------------------------------------------------

def make_result(url, classification, reason, url_final=None, status_code=None,
                content_type=None, redirected=False):
    """Create a standard validation result."""
    return {
        "url_original": url,
        "url_final": url_final,
        "status_code": status_code,
        "content_type": content_type,
        "valid": classification == "valid",
        "classification": classification,
        "reason": reason,
        "redirected": redirected,
    }


# ---------------------------------------------------------------------------
# URL VALIDATION
# ---------------------------------------------------------------------------

def validate_url(url):
    """Validate one URL."""
    if pd.isna(url):
        return make_result(url, "invalid", "missing URL")

    url = str(url).strip()

    if not url:
        return make_result(url, "invalid", "missing URL")

    if not url.startswith(("http://", "https://")):
        return make_result(url, "invalid", "invalid URL scheme")

    session = get_session()

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

    except requests.exceptions.Timeout:
        return make_result(url, "valid", "timeout")

    except requests.exceptions.TooManyRedirects:
        return make_result(url, "valid", "too many redirects")

    except requests.exceptions.SSLError:
        return make_result(url, "valid", "SSL error")

    except requests.exceptions.ConnectionError:
        return make_result(url, "valid", "connection error")

    except requests.exceptions.RequestException as e:
        return make_result(
            url,
            "valid",
            f"request error: {type(e).__name__}",
        )

    final_url = response.url
    status_code = response.status_code
    content_type = response.headers.get("Content-Type", "").lower()

    redirected = final_url.rstrip("/") != url.rstrip("/")

    if status_code in {404, 410}:
        return make_result(
            url,
            "invalid",
            f"HTTP {status_code}",
            url_final=final_url,
            status_code=status_code,
            content_type=content_type,
            redirected=redirected,
        )

    return make_result(
        url,
        "valid",
        f"HTTP {status_code}",
        url_final=final_url,
        status_code=status_code,
        content_type=content_type,
        redirected=redirected,
    )


# ---------------------------------------------------------------------------
# LOAD SOURCE DATA
# ---------------------------------------------------------------------------

def load_source_data():
    """Load source outcome files and collect unique URLs."""
    all_urls = []

    for source, filename in SOURCE_FILES.items():
        input_path = OUTCOME_INPUT_DIR / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found:\n{input_path}")

        df = pd.read_csv(input_path, encoding="utf-8", dtype=str)

        if "url" not in df.columns:
            continue

        urls = df["url"].dropna().astype(str).str.strip()
        urls = urls[urls != ""]
        all_urls.extend(urls.tolist())

    unique_urls = list(dict.fromkeys(all_urls))

    return unique_urls


# ---------------------------------------------------------------------------
# LOAD PREVIOUS VALIDATION
# ---------------------------------------------------------------------------

def load_previous_url_validation():
    """Load the existing URL validation cache."""
    if not VALIDATION_FILE.exists():
        return {}

    previous = pd.read_csv(
        VALIDATION_FILE,
        encoding="utf-8",
        dtype=str,
    )

    required_columns = {"url_original", "classification"}

    missing_columns = required_columns - set(previous.columns)

    if missing_columns:
        raise ValueError(
            "unique_url_validation.csv is missing: "
            + ", ".join(sorted(missing_columns))
        )

    results = {}

    for _, row in previous.iterrows():
        url = str(row["url_original"]).strip()

        if not url:
            continue

        old_classification = str(row["classification"]).strip().lower()

        classification = "invalid" if old_classification == "invalid" else "valid"

        results[url] = {
            "url_original": url,
            "url_final": row.get("url_final", ""),
            "status_code": row.get("status_code", ""),
            "content_type": row.get("content_type", ""),
            "valid": classification == "valid",
            "classification": classification,
            "reason": row.get("reason", ""),
            "redirected": str(row.get("redirected", "")).lower() == "true",
        }

    return results


# ---------------------------------------------------------------------------
# VALIDATE UNIQUE URLS
# ---------------------------------------------------------------------------

def validate_unique_urls(unique_urls, previous_results):
    """Reuse cached results and validate only new URLs."""
    results = {}
    urls_to_validate = []
    reused_count = 0

    for url in unique_urls:
        url = str(url).strip()

        if not url:
            continue

        if url in previous_results:
            results[url] = previous_results[url]
            reused_count += 1
        else:
            urls_to_validate.append(url)

    print()
    print("=" * 70)
    print("URL VALIDATION")
    print("=" * 70)
    print(f"Total unique URLs          : {len(unique_urls):,}")
    print(f"Previously validated       : {reused_count:,}")
    print(f"New URLs to validate       : {len(urls_to_validate):,}")
    print(f"Concurrent workers         : {MAX_WORKERS}")
    print()

    if not urls_to_validate:
        print("No new URLs require validation.")
        return results

    completed = 0
    valid_count = 0
    invalid_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(validate_url, url): url
            for url in urls_to_validate
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]

            try:
                result = future.result()
            except Exception as e:
                result = make_result(
                    url,
                    "valid",
                    f"unexpected error: {type(e).__name__}",
                )

            results[url] = result
            completed += 1

            if result["classification"] == "valid":
                valid_count += 1
            else:
                invalid_count += 1

            if completed % 1000 == 0 or completed == len(urls_to_validate):
                print(
                    f"Checked {completed:,}/{len(urls_to_validate):,} "
                    f"| valid: {valid_count:,} "
                    f"| invalid: {invalid_count:,}"
                )

    return results


# ---------------------------------------------------------------------------
# SAVE VALIDATION RESULTS
# ---------------------------------------------------------------------------

def save_validation_results(results):
    """Save the complete URL validation cache."""
    validation_df = pd.DataFrame(list(results.values()))

    if not validation_df.empty:
        validation_df = validation_df.sort_values("url_original").reset_index(drop=True)

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8",
    )

    return validation_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    unique_urls = load_source_data()
    previous_results = load_previous_url_validation()

    validation_results = validate_unique_urls(
        unique_urls,
        previous_results,
    )

    validation_df = save_validation_results(validation_results)

    valid_count = validation_df["classification"].eq("valid").sum()
    invalid_count = validation_df["classification"].eq("invalid").sum()

    print()
    print("=" * 70)
    print("URL VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Unique URLs              : {len(validation_df):,}")
    print(f"Valid                    : {valid_count:,}")
    print(f"Invalid                  : {invalid_count:,}")
    print()
    print(f"Saved validation cache:\n{VALIDATION_FILE}")


if __name__ == "__main__":
    main()