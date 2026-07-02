import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm
import collect_gtr as gtr
from collections import defaultdict
import re

HEADERS = {
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)"
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

def get_outcome_type(href):
    m = re.search(r"/outcomes/([^/]+)/", href)
    return m.group(1) if m else None

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

    all_outcomes = []
    raw_by_type = defaultdict(list)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Fetching outcomes"):
        outcome_type = get_outcome_type(row.outcome_href)
        if not outcome_type:
            continue

        data = gtr.fetch_json(row.outcome_href, session, delay = args.delay)
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

        for outcome_type, rows in raw_by_type.items():
            df_out = pd.json_normalize(rows, sep=".")
            df_out.to_csv(
                OUTCOME_DIR / f"gtr_{outcome_type}_{timestamp}.csv",
                index=False
            )

            df_out.to_csv(
                OUTCOME_DIR / f"gtr_{outcome_type}_latest.csv",
                index=False
            )

    full_df_out = pd.json_normalize(all_outcomes, sep=".")
    full_df_out.to_csv(OUTCOME_DIR / f"gtr_all_outcomes_{timestamp}.csv")
    full_df_out.to_csv(OUTCOME_DIR / f"gtr_all_outcomes_latest.csv")


    gtr.flush_cache()
    gtr.conn.close()


if __name__ == "__main__":
    main()