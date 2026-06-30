import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timezone
import requests
from tqdm import tqdm
import collect_gtr as gtr

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GTR_DIR = SCRIPT_DIR.parent
DATA_DIR = GTR_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CKPT_DIR = DATA_DIR / "checkpoints"

for d in (RAW_DIR, PROC_DIR, CACHE_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# HTTP status codes that are worth retrying (server-side / transient).
RETRYABLE_STATUS = {500, 502, 503, 504, 429}

# Tunable retry behaviour (overridable from the CLI via globals set in main()).
MAX_RETRIES = 5          # attempts per request before giving up
BACKOFF_BASE = 2.0       # seconds; wait grows 2, 4, 8, 16 ... between attempts


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parse_primary_audience(pa):
    if isinstance(pa, dict):
        return gtr.format_with_pct(pa.get("item", []))
    if isinstance(pa, list):
        return "; ".join(str(x) for x in pa if x)
    if isinstance(pa, str):
        return pa
    return ""


def extract_year(data):
    years = data.get("yearsOfDissemination")
    if years:
        return years.replace(",", "; ")
    raw = (data.get("yearFirstProvided") 
           or data.get("yearsOfDissemination")
           or data.get("datePublished")
           or data.get("start")
           or "")
    if not raw:
        return ""
    raw_str = str(raw).strip()
    # Convert if Unix ms
    if raw_str.isdigit():
        num = int(raw_str)
        # milliseconds
        if num > 10_000_000_000:
            return datetime.fromtimestamp(num / 1000, tz=timezone.utc).year
        # seconds
        if num > 1_000_000_000:
            return datetime.fromtimestamp(num, tz=timezone.utc).year
        # already a year like "2021"
        if len(raw_str) == 4:
            return int(raw_str)
    # ISO date fallback
    try:
        return datetime.fromisoformat(raw[:10]).year
    except Exception:
        return ""
    
def extract_type(data, href):
    """Some outcomes do not have type associated, so take relationship type
    where there is no form or type"""
    # First use explicit fields form the API
    outcome_type = data.get("form") or data.get("type")
    if outcome_type:
        return outcome_type
    
    # Otherwise infer it from the endpoint
    href = href.lower()
    if "collaborations" in href:
        return "Collaboration"
    elif "furtherfundings" in href:
        return "Further funding"
    elif "intellectualproperties" in href:
        return "Intellectual properties"
    return ""

def extract_funding(data):
    amount = data.get("amount") or {}
    formatted_amount = (
        f"{amount.get('currencyCode', '')} {amount.get('amount', '')}"
        if amount else ""
    )
    return formatted_amount


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def parse_outcome(data, href, project_info):
    """Normalise a single GtR outcome into a structured row."""

    return {
        "project_id": project_info["project_id"],
        "grant_reference": project_info["grant_reference"],
        "project_title": project_info["title"],
        "outcome_id": data.get("id", ""),
        "href": href,
        "title": data.get("title") or "",
        "description": gtr.clean_text(data.get("description") or ""),
        "type": extract_type(data, href),
        "journal_title": data.get("journalTitle"),
        "primary_audience": parse_primary_audience(
            data.get("primaryAudience") or data.get("primaryAudiences")),
        "year_of_dissemination": extract_year(data),
        "impact": gtr.clean_text(
            data.get("impact") or data.get("narrative") or data.get("summary") or ""),
        "sector": data.get("sector") or "",
        "funding_id": data.get("fundingId"),
        "funding_amount": extract_funding(data),
        "url": data.get("supportingUrl") or data.get("publicationUrl")
    }



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect CE projects from the UKRI GtR API.")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default 0.3; raise to be gentler)")
    parser.add_argument("--test-limit", type=int, default=None, help="Only process first N rows (for testing)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Retries per request on timeout/server error (default 5)")
    parser.add_argument("--backoff-base", type=float, default=2.0,
                        help="Base seconds for exponential backoff between retries (default 2)")
    args = parser.parse_args()

    global MAX_RETRIES, BACKOFF_BASE
    MAX_RETRIES = max(1, args.max_retries)
    BACKOFF_BASE = max(0.0, args.backoff_base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    file_path = RAW_DIR / "gtr_outcome_hrefs.csv"
    df = pd.read_csv(file_path)
    if args.test_limit:
        df = df.head(args.test_limit)

    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Fetching outcomes"):
        project_info = {
            "project_id": row.project_id,
            "grant_reference": row.grant_reference,
            "title": row.title,
        }

        if not project_info["title"] or not project_info["grant_reference"]:
            continue
        
        href = row.outcome_href
        data = gtr.fetch_json(href, session, delay=args.delay)

        if not data:
            print(f"Failed fetch: {href}")
            continue

        row_out = parse_outcome(data, href, project_info)
        results.append(row_out)

    out = pd.DataFrame(results)

    output_dir = Path(args.out_dir) if args.out_dir else PROC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"gtr_outcomes_{timestamp}.csv"
    latest_path = output_dir / "gtr_outcomes_latest.csv"

    out.to_csv(out_path, index=False, encoding="utf-8")
    out.to_csv(latest_path, index=False, encoding="utf-8")

    print(f"Saved {len(out)} outcomes.")


if __name__ == "__main__":
    main()