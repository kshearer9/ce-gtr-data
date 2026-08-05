"""
test_wos_api.py
===============
Probe for the Web of Science (WoS) Expanded API, run BEFORE building the full
collector. Its only job is to measure, not to collect: for a small handful of
UKRI projects that carry a research-council grant reference, it asks WoS how
many indexed papers acknowledge that grant, and pulls one full record so we can
see the real response shape (funding block + subject categories) before
committing to a schema.

Why a probe first (measure before you build):
  - WoS holds PUBLICATIONS, not GtR projects. There is no project record to
    fetch; we find the papers that acknowledge each grant.
  - Only ~337 of the 1,380 projects carry a council-format reference
    (e.g. EP/V042432/1). The rest are Innovate UK / Horizon numeric IDs that
    WoS funding search will not recognise. This probe quantifies the real hit
    rate on the matchable subset before we invest in the full pipeline.

Two funding field tags are tested per project, both cheaply (count=0 returns the
match count without pulling records):
  FG = grant number      e.g.  FG=(EP/V042432/1)
  FT = funding text      e.g.  FT=(EP/V042432/1)
The comparison tells us which tag is the reliable join key.

This is a throwaway diagnostic. It writes nothing except console output; keep it
in the repo as documentation of how the matching decision was made, or delete it
once collect_wos.py is settled.

Run from the project root:
    python3 -m scripts.collection.test_wos_api
    python3 -m scripts.collection.test_wos_api --n 8

Requires WOS_API_KEY in a local .env file (see README > Setup > API keys).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a hard requirement, fail with a clear message
    sys.exit("python-dotenv is not installed. Run: pip install python-dotenv")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# WoS Expanded API base endpoint (item-level metadata: funding, categories,
# times-cited). NB this differs from the Starter endpoint.
WOS_BASE_URL = "https://wos-api.clarivate.com/api/wos"

# Core Collection is the database that indexes journal articles and their
# funding acknowledgements. (GRANTS is a separate funder-side database and is
# NOT what we want here.)
DATABASE_ID = "WOS"

# Council-format grant reference, e.g. EP/V042432/1, NE/V01076X/1, EP/Y53058X/1.
# Two letters / seven alphanumerics / trailing revision digits. These are the
# references WoS funding search can plausibly match; numeric Innovate UK / Horizon
# IDs are excluded because WoS will not recognise them.
COUNCIL_REF_PATTERN = re.compile(r"^[A-Z]{2}/[A-Z0-9]{7}/\d+$")

# Default input: the enriched, cleaned project dataset. grant_reference is 100%
# populated here. (Kirsty's merged projects_-_cleaned.csv carries the same column
# and can be swapped in once it is the canonical input.)
DEFAULT_INPUT = Path("data/cleaned/merged/projects_-_cleaned.csv")
# Fallback for local runs before the merge lands / if the path differs.
FALLBACK_INPUTS = [
    Path("data/processed/gtr_ce_projects_enriched_clean.csv"),
    Path("data/cleaned/merged/projects_cleaned.csv"),
]

# Retry behaviour, mirrored from collect_gtr.py for consistency.
RETRYABLE_STATUS = {500, 502, 503, 504, 429}
MAX_RETRIES = 5
BACKOFF_BASE = 2.0

# WoS Expanded free/basic plans are rate-limited (typically ~2 req/s). Be polite.
REQUEST_DELAY = 1.0


# --------------------------------------------------------------------------- #
# HTTP helper (retry + backoff), same shape as collect_gtr._request_with_retries
# --------------------------------------------------------------------------- #

def request_with_retries(session, params):
    """GET the WoS endpoint with retry-and-backoff on transient failures.

    Returns parsed JSON on success. Raises the last exception if every attempt
    fails, so the caller can decide whether to skip or abort.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(WOS_BASE_URL, params=params, timeout=60)
            if resp.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(
                    f"{resp.status_code} (retryable) for {resp.url}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = (
                isinstance(exc, (requests.Timeout, requests.ConnectionError))
                or status in RETRYABLE_STATUS
            )
            last_exc = exc
            if not retryable or attempt == MAX_RETRIES:
                raise
            if status == 429:
                retry_after = exc.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else BACKOFF_BASE * (2 ** (attempt - 1))
            else:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"    (attempt {attempt}/{MAX_RETRIES} failed: {exc}; "
                  f"retrying in {wait:.0f}s)", flush=True)
            time.sleep(wait)
    raise last_exc


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #

def count_matches(session, field_tag, grant_ref):
    """Return the number of WoS records matching a funding query, cheaply.

    count=0 asks the API for the RecordsFound total without returning any
    records, so this costs one lightweight request per call.
    """
    query = f"{field_tag}=({grant_ref})"
    params = {
        "databaseId": DATABASE_ID,
        "usrQuery": query,
        "count": 0,
        "firstRecord": 1,
    }
    data = request_with_retries(session, params)
    return data["QueryResult"]["RecordsFound"]


def fetch_one_record(session, field_tag, grant_ref):
    """Pull a single full record for a query, so we can inspect its shape."""
    query = f"{field_tag}=({grant_ref})"
    params = {
        "databaseId": DATABASE_ID,
        "usrQuery": query,
        "count": 1,
        "firstRecord": 1,
    }
    data = request_with_retries(session, params)
    records = data.get("Data", {}).get("Records", {}).get("records", {}).get("REC", [])
    return records[0] if records else None


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #

def resolve_input_path(explicit):
    """Pick the project list: explicit arg, then default, then fallbacks."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"Input file not found: {p}")
        return p
    for candidate in [DEFAULT_INPUT, *FALLBACK_INPUTS]:
        if candidate.exists():
            return candidate
    sys.exit(
        "No project list found. Pass one with --input, or place the cleaned "
        f"dataset at {DEFAULT_INPUT}."
    )


def load_council_refs(path, n):
    """Read the first n council-format grant references from the project list.

    Uses only the stdlib csv module so the probe has no pandas dependency.
    Returns a list of (project_id, grant_reference, lead_funder, title) tuples.
    """
    import csv

    picked = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Column names differ slightly between the enriched file and Kirsty's
        # merged file; probe for whichever title column is present.
        title_col = next((c for c in ("title", "title_clean") if c in reader.fieldnames), None)
        for row in reader:
            ref = (row.get("grant_reference") or "").strip()
            if COUNCIL_REF_PATTERN.match(ref):
                picked.append((
                    row.get("project_id", ""),
                    ref,
                    row.get("lead_funder", ""),
                    (row.get(title_col) or "")[:70] if title_col else "",
                ))
            if len(picked) >= n:
                break
    return picked


# --------------------------------------------------------------------------- #
# Record inspection (what does a real WoS record look like?)
# --------------------------------------------------------------------------- #

def summarise_record(rec):
    """Print the fields we care about from one WoS record, defensively.

    The Expanded API response is deeply nested and varies by record, so every
    access is guarded. This is exactly the shape we will formalise into the
    collector's schema once we have seen a few real examples.
    """
    if rec is None:
        print("    (no record returned to inspect)")
        return

    def dig(d, *keys, default="-"):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    uid = rec.get("UID", "-")

    static = dig(rec, "static_data", default={})
    summary = dig(static, "summary", default={})

    # Title + source live under summary.titles.title as a list of typed entries.
    titles = dig(summary, "titles", "title", default=[])
    if isinstance(titles, dict):
        titles = [titles]
    item_title = next((t.get("content") for t in titles if t.get("type") == "item"), "-")
    source_title = next((t.get("content") for t in titles if t.get("type") == "source"), "-")

    pub_year = dig(summary, "pub_info", "pubyear", default="-")

    # Times cited lives in dynamic_data.
    tc = dig(rec, "dynamic_data", "citation_related", "tc_list", "silo_tc", default={})
    times_cited = tc.get("local_count", "-") if isinstance(tc, dict) else "-"

    # Funding block (the whole point of route a).
    fund = dig(static, "fullrecord_metadata", "fund_ack", default={})
    grants = dig(fund, "grants", "grant", default=[])
    if isinstance(grants, dict):
        grants = [grants]
    grant_agencies = []
    for g in grants:
        agency = g.get("grant_agency", "-")
        ids = g.get("grant_ids", {})
        gid = ids.get("grant_id", "-") if isinstance(ids, dict) else "-"
        grant_agencies.append(f"{agency} [{gid}]")

    # Subject categories (the QUARANTINED third taxonomy; collected, not used
    # in the discipline crosswalk).
    subjects = dig(static, "fullrecord_metadata", "category_info", "subjects", "subject", default=[])
    if isinstance(subjects, dict):
        subjects = [subjects]
    cats = [s.get("content", "-") for s in subjects if isinstance(s, dict)]

    print(f"    UID:          {uid}")
    print(f"    Title:        {item_title[:80]}")
    print(f"    Source:       {source_title[:60]}  ({pub_year})")
    print(f"    Times cited:  {times_cited}")
    print(f"    Funding:      {'; '.join(grant_agencies[:4]) or '-'}")
    print(f"    Categories:   {', '.join(cats[:6]) or '-'}  [QUARANTINED]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Probe the WoS Expanded API.")
    parser.add_argument("--n", type=int, default=6,
                        help="How many council-format projects to probe (default 6).")
    parser.add_argument("--input", default=None,
                        help="Path to the project list CSV (defaults to the cleaned dataset).")
    parser.add_argument("--inspect", type=int, default=2,
                        help="How many of the matched projects to pull a full record for (default 2).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("WOS_API_KEY")
    if not api_key:
        sys.exit(
            "WOS_API_KEY not found. Create a local .env file in the project root "
            "with WOS_API_KEY=your_key (see README > Setup > API keys)."
        )

    input_path = resolve_input_path(args.input)
    refs = load_council_refs(input_path, args.n)
    if not refs:
        sys.exit(f"No council-format grant references found in {input_path}.")

    print("=" * 74)
    print("WoS Expanded API probe")
    print(f"Input:    {input_path}")
    print(f"Projects: {len(refs)} council-format references")
    print(f"Endpoint: {WOS_BASE_URL}  (db={DATABASE_ID})")
    print("=" * 74)

    session = requests.Session()
    session.headers.update({"X-ApiKey": api_key, "Accept": "application/json"})

    # A quick auth sanity check on the first reference so a bad key fails loudly
    # and immediately rather than after several confusing errors.
    try:
        _ = count_matches(session, "FG", refs[0][1])
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            sys.exit(f"Auth failed ({status}). Check WOS_API_KEY is a valid Expanded key.")
        raise

    print(f"\n{'project_id':<38} {'grant_ref':<15} {'FG':>5} {'FT':>5}  funder")
    print("-" * 74)

    results = []
    for project_id, ref, funder, title in refs:
        fg = count_matches(session, "FG", ref)
        time.sleep(REQUEST_DELAY)
        ft = count_matches(session, "FT", ref)
        time.sleep(REQUEST_DELAY)
        results.append((project_id, ref, funder, title, fg, ft))
        print(f"{project_id:<38} {ref:<15} {fg:>5} {ft:>5}  {funder}")

    # Summary: which tag is the better join key, and overall hit rate.
    fg_hits = sum(1 for r in results if r[4] > 0)
    ft_hits = sum(1 for r in results if r[5] > 0)
    any_hits = sum(1 for r in results if r[4] > 0 or r[5] > 0)
    print("-" * 74)
    print(f"Projects with >=1 match:  FG {fg_hits}/{len(results)}   "
          f"FT {ft_hits}/{len(results)}   either {any_hits}/{len(results)}")

    # Inspect a couple of full records so we can see the real response shape.
    to_inspect = [r for r in results if r[4] > 0 or r[5] > 0][:args.inspect]
    if to_inspect:
        print("\n" + "=" * 74)
        print("Full-record inspection (schema design for collect_wos.py)")
        print("=" * 74)
        for project_id, ref, funder, title, fg, ft in to_inspect:
            tag = "FG" if fg > 0 else "FT"
            print(f"\n  {project_id}  ({ref}, via {tag}=):")
            rec = fetch_one_record(session, tag, ref)
            time.sleep(REQUEST_DELAY)
            summarise_record(rec)

    print("\nProbe complete. Nothing was written to disk.")


if __name__ == "__main__":
    main()