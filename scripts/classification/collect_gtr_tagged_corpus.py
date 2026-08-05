"""Collect a broad corpus of subject-tagged UKRI projects for classifier training.

Modelled on scripts/collection/collect_gtr_projects.py: same API headers,
retry-and-backoff, SQLite response cache and JSONL raw output. Differences:
it paginates the WHOLE GtR projects index (no search terms, no CE screening),
keeps only projects that carry a researchSubjects classification, and excludes
every project in the CE dataset so the training corpus is disjoint from the
evaluation set.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/collect_gtr_tagged_corpus.py

Options:
    --target-tagged N   stop once N tagged projects are collected (default 30000)
    --max-pages N       hard page cap (safety; default none)
    --fresh             ignore the page checkpoint and start from page 1

Outputs:
    data/classification/gtr_tagged_corpus.csv        (the training corpus)
    data/raw/gtr_corpus_raw.jsonl                    (raw kept records, append)
    checkpoints/gtr_corpus_checkpoint.json           (page number only)
Cache:  cache/gtr_corpus_cache.db (separate DB; does not touch the CE cache)

Rows are appended to the CSV as they are collected, and the checkpoint stores
only the next page number, so an interrupted run resumes cheaply and memory
use stays flat regardless of corpus size.
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = DATA_DIR / "classification"
RAW_PATH = DATA_DIR / "raw" / "gtr_corpus_raw.jsonl"
CKPT_PATH = ROOT_DIR / "checkpoints" / "gtr_corpus_checkpoint.json"
CACHE_DB = ROOT_DIR / "cache" / "gtr_corpus_cache.db"
CE_PROJECTS = DATA_DIR / "cleaned" / "merged" / "projects.csv"

BASE_URL = "https://gtr.ukri.org/gtr/api/projects"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
}
PAGE_SIZE = 100
REQUEST_DELAY = 0.2
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


# --------------------------------------------------------------------------
# Cache (same pattern as the CE collector, separate database)
# --------------------------------------------------------------------------

conn = sqlite3.connect(CACHE_DB)
conn.execute(
    "CREATE TABLE IF NOT EXISTS api_cache (url TEXT PRIMARY KEY, response TEXT NOT NULL)"
)
conn.commit()


def get_cache(url):
    row = conn.execute(
        "SELECT response FROM api_cache WHERE url = ?", (url,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def save_cache(url, data):
    conn.execute(
        "INSERT OR REPLACE INTO api_cache (url, response) VALUES (?, ?)",
        (url, json.dumps(data, ensure_ascii=False)),
    )
    conn.commit()


# --------------------------------------------------------------------------
# Requests (identical retry semantics to the CE collector)
# --------------------------------------------------------------------------

def request_with_retries(session, url, params=None):
    retryable_status = {500, 502, 503, 504, 429}
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=60)
            if resp.status_code in retryable_status:
                raise requests.HTTPError(
                    f"{resp.status_code} (retryable) for {resp.url}", response=resp
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = (
                isinstance(exc, (requests.Timeout, requests.ConnectionError))
                or status in retryable_status
            )
            last_exc = exc
            if not retryable or attempt == MAX_RETRIES:
                raise
            if status == 429 and exc.response.headers.get("Retry-After"):
                wait = float(exc.response.headers["Retry-After"])
            else:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"\n    (attempt {attempt}/{MAX_RETRIES} failed: {exc}; "
                  f"retrying in {wait:.0f}s)", flush=True)
            time.sleep(wait)
    raise last_exc


def fetch_page(session, page):
    url_key = f"{BASE_URL}?p={page}&s={PAGE_SIZE}"
    cached = get_cache(url_key)
    if cached is not None:
        return cached
    data = request_with_retries(session, BASE_URL, params={"p": page, "s": PAGE_SIZE})
    time.sleep(REQUEST_DELAY)
    save_cache(url_key, data)
    return data


# --------------------------------------------------------------------------
# Flattening (subset of the CE collector's flatten_project)
# --------------------------------------------------------------------------

def format_with_pct(items):
    parts = []
    for item in items:
        text = (item.get("text") or "").strip()
        pct = item.get("percentage")
        if text and pct is not None:
            parts.append(f"{text} ({pct}%)")
        elif text:
            parts.append(text)
    return "; ".join(parts)


def flatten(project):
    subjects = project.get("researchSubjects", {}).get("researchSubject", [])
    identifiers = project.get("identifiers", {}).get("identifier", [])
    grant_ref = next(
        (i.get("value", "") for i in identifiers if i.get("type") == "RCUK"), ""
    )
    fund = project.get("fund", {}) or {}
    funder = (fund.get("funder", {}) or {}).get("name", "") or \
        project.get("leadFunder", "")
    return {
        "project_id": project.get("id", ""),
        "grant_reference": grant_ref,
        "title": (project.get("title") or "").strip(),
        "abstract_text": (project.get("abstractText") or "").strip(),
        "research_subjects": format_with_pct(subjects),
        "lead_funder": funder,
        "grant_category": project.get("grantCategory", ""),
        "status": project.get("status", ""),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tagged", type=int, default=30000)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CE_PROJECTS.exists():
        sys.exit(f"CE dataset not found at {CE_PROJECTS}; run from the repo root.")
    ce_ids = set(pd.read_csv(CE_PROJECTS, usecols=["project_id"]).project_id)
    print(f"Excluding {len(ce_ids)} CE projects from the corpus "
          "(keeps training disjoint from evaluation).")

    out = OUT_DIR / "gtr_tagged_corpus.csv"
    COLUMNS = ["project_id", "grant_reference", "title", "abstract_text",
               "research_subjects", "lead_funder", "grant_category", "status"]

    start_page = 1
    kept = 0
    if args.fresh:
        for path in (out, RAW_PATH):
            if path.exists():
                path.unlink()
        if CKPT_PATH.exists():
            CKPT_PATH.unlink()
    elif CKPT_PATH.exists():
        start_page = json.loads(CKPT_PATH.read_text())["next_page"]
        if out.exists():
            kept = sum(1 for _ in open(out, encoding="utf-8")) - 1
        print(f"Resuming from page {start_page}; {kept} tagged rows already on disk.")

    if not out.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)

    session = requests.Session()
    first = fetch_page(session, start_page)
    total_pages = first.get("totalPages", 1)
    total_size = first.get("totalSize", 0)
    end_page = min(total_pages, args.max_pages) if args.max_pages else total_pages
    print(f"GtR index: {total_size} projects, {total_pages} pages "
          f"(sweeping up to page {end_page}, page size {PAGE_SIZE}).")

    scanned = 0
    t0 = time.time()
    page = start_page
    with open(RAW_PATH, "a", encoding="utf-8") as raw:
        while page <= end_page and kept < args.target_tagged:
            data = first if page == start_page else fetch_page(session, page)
            batch = []
            for proj in data.get("project", []):
                scanned += 1
                if proj.get("id") in ce_ids:
                    continue
                if not proj.get("researchSubjects", {}).get("researchSubject", []):
                    continue
                row = flatten(proj)
                if not row["title"] and not row["abstract_text"]:
                    continue
                raw.write(json.dumps(proj, ensure_ascii=False) + "\n")
                batch.append(row)
            if batch:
                pd.DataFrame(batch, columns=COLUMNS).to_csv(
                    out, mode="a", header=False, index=False)
                kept += len(batch)
            if page % 25 == 0:
                raw.flush()
                CKPT_PATH.write_text(json.dumps({"next_page": page + 1}))
                print(f"  page {page}/{end_page} | tagged {kept} "
                      f"({kept / max(scanned, 1):.0%} of scanned) | "
                      f"{time.time() - t0:.0f}s", flush=True)
            page += 1
    CKPT_PATH.write_text(json.dumps({"next_page": page}))

    df = pd.read_csv(out).drop_duplicates(subset="project_id")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} tagged projects to {out}")
    print(f"Scanned {scanned} projects this run; "
          f"finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")
    print(f"Abstract present for {(df.abstract_text.fillna('').str.len() > 0).mean():.0%} "
          f"of the corpus.")


if __name__ == "__main__":
    main()
