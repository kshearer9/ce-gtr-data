import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm
import collect_gtr as gtr
from collections import defaultdict
import re
import sqlite3
import json
import time

HEADERS = {
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
    "Accept": "application/json"
}

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
GTR_DIR = SCRIPT_DIR.parent
DATA_DIR = GTR_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUTCOME_DIR = PROC_DIR / "outcomes"

CACHE_DIR = GTR_DIR / "cache"
CKPT_DIR = DATA_DIR / "checkpoints"

for d in (RAW_DIR, PROC_DIR, CACHE_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

CACHE_DB = CACHE_DIR / "gtr_outcome_cache.db"
conn = sqlite3.connect(CACHE_DB)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS outcome_cache (
    href TEXT PRIMARY KEY,
    response TEXT
)
""")
conn.commit()


# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------

def normalise_key(key: str) -> str:
    return key.strip()


def get_cache(href):
    href = href.strip()
    cursor.execute(
        "SELECT response FROM outcome_cache WHERE href = ?",
        (href,)
    )
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def save_cache(href, data):
    href = href.strip()
    cursor.execute(
        f"""
        INSERT OR REPLACE INTO outcome_cache (href, response)
        VALUES (?, ?)
        """,
        (href, json.dumps(data))
    )
    conn.commit()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_outcome_type(href):
    m = re.search(r"/outcomes/([^/]+)/", href)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# API FETCHERS
# ---------------------------------------------------------------------------

def fetch_json(href, session, delay, max_retries, backoff_base, failed):
    """Generic helper: fetch any GtR API URL and return parsed JSON.
    Retries transient failures; returns {} only if all retries are exhausted,
    so enrichment of a single project can fail softly without killing the run."""
    if not href:
        return {}

    # Check SQLite cache first
    cached = get_cache(href)
    if cached is not None:
        return cached
    
    try:
        data = gtr._request_with_retries(session, href, headers=HEADERS, max_retries=max_retries, backoff_base=backoff_base)
        time.sleep(delay)
        # Save to cache
        save_cache(href, data)
        return data
    
    except requests.RequestException as e:
        failed[href] = str(e)
        return {}
    
def retry_failed(failed, session, args):
    """Retry failed URLs stored in dict form."""
    still_failed = {}
    for href in tqdm(list(failed.keys()), desc="Retrying failed requests"):
        data = fetch_json(
            href,
            session,
            delay=args.delay,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            failed=still_failed
        )
        # if still failing, it stays in still_failed
        if data:
            print(f"Recovered: {href}")
    return still_failed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect CE projects from the UKRI GtR API.")
    parser.add_argument("--delay", type=float, default=1,
                        help="Seconds between API calls (default 0.3; raise to be gentler)")
    parser.add_argument("--test-limit", type=int, default=None, help="Only process first N rows (for testing)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Retries per request on timeout/server error (default 5)")
    parser.add_argument("--backoff-base", type=float, default=2.0,
                        help="Base seconds for exponential backoff between retries (default 2)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    file_path = RAW_DIR / "gtr_outcome_hrefs.csv"
    df = pd.read_csv(file_path, encoding = "utf-8")
    if args.test_limit:
        df = df.head(args.test_limit)

    all_outcomes = []
    failed = {}
    raw_by_type = defaultdict(list)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Fetching outcomes"):
        outcome_type = get_outcome_type(row.outcome_href)
        if not outcome_type:
            continue

        data = fetch_json(row.outcome_href, session, delay=args.delay, 
                          max_retries=args.max_retries, 
                          backoff_base=args.backoff_base, failed=failed)
        if not data:
            continue

        outcome = {
            "project_id": row.project_id,
            "grant_reference": row.grant_reference,
            "project_title": row.title,
            "href": row.outcome_href,
            "outcome_type": outcome_type,
            **data
        }

        raw_by_type[outcome_type].append(outcome)
        all_outcomes.append(outcome)
        if len(all_outcomes) % 100 == 0:
            conn.commit()

    for outcome_type, rows in raw_by_type.items():
        df_out = pd.json_normalize(rows, sep=".")
        df_out.to_csv(
            OUTCOME_DIR / f"gtr_{outcome_type}_{timestamp}.csv",
            index=False, encoding="utf-8"
        )

        df_out.to_csv(
            OUTCOME_DIR / f"gtr_{outcome_type}_latest.csv",
            index=False, encoding="utf-8"
        )

    full_df_out = pd.json_normalize(all_outcomes, sep=".")
    full_df_out.to_csv(OUTCOME_DIR / f"gtr_all_outcomes_{timestamp}.csv", 
                       index=False, encoding="utf-8")
    full_df_out.to_csv(OUTCOME_DIR / f"gtr_all_outcomes_latest.csv",
                       index=False, encoding="utf-8")
    
    if failed:
        print(f"\nRetrying {len(failed)} failed requests...")
        failed = retry_failed(failed, session, args)
        print(f"Still failed after retry: {len(failed)}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()