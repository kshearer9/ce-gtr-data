"""
collect_wos.py
==============
Retrieve publication outcomes for UKRI Gateway to Research projects using the
Web of Science (WoS) Expanded API.

Pipeline:
    1. Builds a WoS search query from the project grant reference (FG= tag).
    2. Searches the WoS Core Collection for papers acknowledging that grant.
    3. Retrieves full item-level metadata for each matched outcome.
    4. Extracts outcome, author, institution, funding and subject metadata.
    5. Saves raw API responses and processed datasets.
    6. Optionally extracts cited references into a separate table.

Matching route (validated before building, see test_wos_api.py):
    WoS indexes the grant numbers quoted in funding acknowledgements. UKRI
    research-council references are self-identifying (e.g. EP/V042432/1), so
    FG=(reference) reliably links a grant to its output papers. A six-project
    probe returned matches for 6/6, with FG equalling or exceeding the free-text
    FT= tag on every project, so FG is used as the primary join key.

    Only research-council-format references are queried. Innovate UK and Horizon
    Europe Guarantee projects carry bare numeric UKRI internal identifiers
    (e.g. 10043489, 101901) which are not indexed as grant numbers in WoS: a test
    query for a Horizon reference returned zero records. Querying them would
    spend quota to confirm zeros, so they are excluded by design and the
    exclusion is reported in the run summary for transparency.

Citation counts:
    dynamic_data.citation_related.tc_list.silo_tc is a LIST of per-database
    entries, not a single value. The Core Collection count (the standard
    bibliometric "Times Cited") is the entry with coll_id == "WOS". The
    all-database total is coll_id == "WOK". Both are captured.

Exported Outputs (in data/processed/wos/):
- wos_outcomes_latest.csv - one row per project-paper pair (the attribution
  unit for inputs-vs-outputs analysis). A paper acknowledging two in-scope
  grants correctly appears once per project.
- wos_outcomes_unique_latest.csv - deduplicated paper-level table, so corpus
  totals are not inflated by multi-grant papers.
- wos_outcomes_institutions_latest.csv - author affiliation institutions
  associated with each matched outcome.
- Optional: wos_outcomes_references_latest.csv - bibliographic metadata for
  references cited by each matched outcome (--references).

Resuming:
    Progress is checkpointed after every --checkpoint-every projects. Re-running
    the same command reloads the checkpoint and skips completed projects. Use
    --fresh to ignore any checkpoint and start over.

Run examples (from the project root):
    python -m scripts.collection.collect_wos --limit 5      (quick test)
    python -m scripts.collection.collect_wos                (full run)
    python -m scripts.collection.collect_wos --references   (full run + refs)
    python -m scripts.collection.collect_wos --fresh        (ignore checkpoint)

Note: use `python` rather than `python3` if your Anaconda interpreter holds the
project dependencies.

Requires WOS_API_KEY in a local .env file (see README > Setup > API keys).
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv is not installed. Run: pip install python-dotenv")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

CLEAN_INPUT_DIR = DATA_DIR / "cleaned"
PROC_INPUT_DIR = DATA_DIR / "processed" / "gtr"
RAW_DIR = DATA_DIR / "raw" / "wos"
PROC_DIR = DATA_DIR / "processed" / "wos"

CACHE_DIR = ROOT_DIR / "cache"
CKPT_DIR = ROOT_DIR / "checkpoints"

for d in (RAW_DIR, PROC_DIR, CACHE_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Candidate input files, tried in order. The merged team dataset is preferred
# once available; the enriched cleaned file is the fallback.
INPUT_CANDIDATES = [
    CLEAN_INPUT_DIR / "merged" / "projects.csv",
    CLEAN_INPUT_DIR / "gtr_projects_clean.csv",
    CLEAN_INPUT_DIR / "merged" / "projects.csv",
    CLEAN_INPUT_DIR / "merged" / "projects.csv",
    PROC_INPUT_DIR / "gtr_ce_projects_enriched_clean.csv",
    DATA_DIR / "processed" / "gtr_ce_projects_enriched_clean.csv",
]


# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------

BASE_URL = "https://wos-api.clarivate.com/api/wos"
REFERENCES_URL = "https://wos-api.clarivate.com/api/wos/references"
DATABASE_ID = "WOS"

# The API returns at most 100 records per request.
PAGE_SIZE = 100

# Observed allowance: x-req-reqpersec-remaining = 2. Stay comfortably inside it.
REQUEST_DELAY = 0.6

# Retry behaviour, mirrored from collect_gtr_projects.py for consistency.
RETRYABLE_STATUS = {500, 502, 503, 504, 429}
MAX_RETRIES = 5
BACKOFF_BASE = 2.0

# Research-council grant reference format, e.g. EP/V042432/1, NE/V01076X/1.
# These are the references WoS indexes as grant numbers.
COUNCIL_REF_PATTERN = re.compile(r"^[A-Z]{2}/[A-Z0-9]{7}/\d+$")


# ---------------------------------------------------------------------------
# CACHE SETUP
# ---------------------------------------------------------------------------

CACHE_DB = CACHE_DIR / "wos_cache.db"
SEARCH_PREFIX = "SEARCH::"
REFS_PREFIX = "REFS::"

conn = sqlite3.connect(CACHE_DB)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS search_cache (
    query TEXT PRIMARY KEY,
    response TEXT
)
""")
conn.commit()


def cache_get(key):
    """Return a cached API response, or None if not present."""
    cursor.execute("SELECT response FROM search_cache WHERE query = ?", (key,))
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def cache_put(key, payload):
    """Store an API response against a cache key."""
    cursor.execute(
        "INSERT OR REPLACE INTO search_cache (query, response) VALUES (?, ?)",
        (key, json.dumps(payload)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def request_with_retries(session, url, params):
    """GET a WoS endpoint with retry-and-backoff on transient failures.

    Returns parsed JSON on success. Raises the last exception if every attempt
    fails, so the caller can decide whether to skip the project or abort.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
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
            print(f"\n    (attempt {attempt}/{MAX_RETRIES} failed: {exc}; "
                  f"retrying in {wait:.0f}s)", flush=True)
            time.sleep(wait)
    raise last_exc


# ---------------------------------------------------------------------------
# PARSING HELPERS
#
# The Expanded API collapses single-item collections into dicts and expands
# multi-item ones into lists, so every accessor has to tolerate both shapes.
# ---------------------------------------------------------------------------

def as_list(node):
    """Normalise a dict-or-list-or-None node into a list."""
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]


def dig(node, *keys, default=None):
    """Walk a nested structure by key, returning default if any step is absent."""
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def get_titles(rec):
    """Return (item_title, source_title) from the typed titles list."""
    titles = as_list(dig(rec, "static_data", "summary", "titles", "title", default=[]))
    item = next((t.get("content") for t in titles
                 if isinstance(t, dict) and t.get("type") == "item"), None)
    source = next((t.get("content") for t in titles
                   if isinstance(t, dict) and t.get("type") == "source"), None)
    return item, source


def get_identifiers(rec):
    """Return a dict of identifier type -> value (doi, issn, eissn, ...)."""
    ids = as_list(dig(rec, "dynamic_data", "cluster_related", "identifiers",
                      "identifier", default=[]))
    out = {}
    for entry in ids:
        if isinstance(entry, dict) and entry.get("type"):
            # Keep the first value seen for each type.
            out.setdefault(entry["type"], entry.get("value"))
    return out


def get_times_cited(rec):
    """Return (core_collection_count, all_databases_count).

    tc_list.silo_tc is a list of per-database entries. coll_id 'WOS' is the
    Core Collection figure used as the standard Times Cited; 'WOK' is the
    all-database total.
    """
    silos = as_list(dig(rec, "dynamic_data", "citation_related", "tc_list",
                        "silo_tc", default=[]))
    core = all_db = None
    for silo in silos:
        if not isinstance(silo, dict):
            continue
        if silo.get("coll_id") == "WOS":
            core = silo.get("local_count")
        elif silo.get("coll_id") == "WOK":
            all_db = silo.get("local_count")
    return core, all_db


def get_subject_categories(rec):
    """Return (traditional_categories, extended_categories) as pipe-joined str.

    These are the WoS subject categories. They form a THIRD taxonomy alongside
    GtR research subjects and OpenAlex fields, so they are collected but
    QUARANTINED: they are not used in the discipline crosswalk unless that
    decision is revisited.
    """
    subjects = as_list(dig(rec, "static_data", "fullrecord_metadata",
                           "category_info", "subjects", "subject", default=[]))
    traditional, extended = [], []
    for s in subjects:
        if not isinstance(s, dict):
            continue
        content = s.get("content")
        if not content:
            continue
        if s.get("ascatype") == "traditional":
            traditional.append(content)
        else:
            extended.append(content)
    return ";".join(traditional), ";".join(extended)


def get_citation_topics(rec):
    """Return (macro, meso, micro) from Clarivate's Citation Topics scheme.

    Also a supplementary taxonomy, collected for interest and QUARANTINED from
    the discipline crosswalk.
    """
    subjects = as_list(dig(rec, "dynamic_data", "citation_related",
                           "citation_topics", "subj-group", "subject", default=[]))
    levels = {}
    for s in subjects:
        if isinstance(s, dict) and s.get("content-type"):
            levels[s["content-type"]] = s.get("content")
    return levels.get("macro"), levels.get("meso"), levels.get("micro")


def get_sdgs(rec):
    """Return pipe-joined UN Sustainable Development Goal categories."""
    cats = as_list(dig(rec, "dynamic_data", "citation_related", "SDG",
                       "sdg_category", default=[]))
    return ";".join(c.get("content", "") for c in cats if isinstance(c, dict))


def get_keywords(rec):
    """Return (author_keywords, keywords_plus) as pipe-joined strings."""
    author_kw = as_list(dig(rec, "static_data", "fullrecord_metadata",
                            "keywords", "keyword", default=[]))
    plus_kw = as_list(dig(rec, "static_data", "item", "keywords_plus",
                          "keyword", default=[]))
    # Entries may be plain strings or {'content': ...} dicts.
    def flatten(items):
        out = []
        for k in items:
            if isinstance(k, str):
                out.append(k)
            elif isinstance(k, dict) and k.get("content"):
                out.append(str(k["content"]))
        return ";".join(out)
    return flatten(author_kw), flatten(plus_kw)


def get_abstract(rec):
    """Return the abstract text, joining multi-paragraph abstracts.

    Note: the Expanded API does return abstracts. They are collected for
    completeness, but GtR project abstracts remain the text source for the
    topic-modelling and classification work, to avoid mixing input-side and
    output-side text.
    """
    paras = as_list(dig(rec, "static_data", "fullrecord_metadata", "abstracts",
                        "abstract", "abstract_text", "p", default=[]))
    out = []
    for p in paras:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict) and p.get("content"):
            out.append(str(p["content"]))
    return " ".join(out)


def get_funding(rec):
    """Return (agencies, grant_ids, funding_text) as pipe-joined strings.

    A paper may acknowledge several grants from several funders. All are kept,
    since multi-grant attribution matters for inputs-vs-outputs analysis.
    """
    grants = as_list(dig(rec, "static_data", "fullrecord_metadata", "fund_ack",
                         "grants", "grant", default=[]))
    agencies, ids = [], []
    for g in grants:
        if not isinstance(g, dict):
            continue
        if g.get("grant_agency"):
            agencies.append(str(g["grant_agency"]))
        raw_ids = dig(g, "grant_ids", "grant_id", default=[])
        for gid in as_list(raw_ids):
            if isinstance(gid, str):
                ids.append(gid)
            elif isinstance(gid, dict) and gid.get("content"):
                ids.append(str(gid["content"]))
    fund_text_parts = as_list(dig(rec, "static_data", "fullrecord_metadata",
                                  "fund_ack", "fund_text", "p", default=[]))
    fund_text = " ".join(
        p if isinstance(p, str) else str(p.get("content", ""))
        for p in fund_text_parts if p
    )
    return ";".join(agencies), ";".join(ids), fund_text


def get_authors(rec):
    """Return (author_names, researcher_ids, orcids) as pipe-joined strings."""
    names = as_list(dig(rec, "static_data", "summary", "names", "name", default=[]))
    display, rids, orcids = [], [], []
    for n in names:
        if not isinstance(n, dict):
            continue
        if n.get("role") and n["role"] != "author":
            continue
        if n.get("display_name"):
            display.append(str(n["display_name"]))
        elif n.get("full_name"):
            display.append(str(n["full_name"]))
        if n.get("r_id"):
            rids.append(str(n["r_id"]))
        if n.get("orcid_id"):
            orcids.append(str(n["orcid_id"]))
    return ";".join(display), ";".join(rids), ";".join(orcids)


def get_institutions(rec):
    """Return a list of institution dicts for the affiliations file.

    Prefers the Clarivate-preferred organisation name where present, since
    that is the normalised form and is more reliable for institution-level
    aggregation than the raw address string.
    """
    addresses = as_list(dig(rec, "static_data", "fullrecord_metadata",
                            "addresses", "address_name", default=[]))
    rows = []
    for entry in addresses:
        spec = dig(entry, "address_spec", default={}) if isinstance(entry, dict) else {}
        if not isinstance(spec, dict):
            continue
        orgs = as_list(dig(spec, "organizations", "organization", default=[]))
        preferred, raw = None, None
        for o in orgs:
            if isinstance(o, dict):
                if o.get("pref") == "Y":
                    preferred = o.get("content")
                else:
                    raw = raw or o.get("content")
            elif isinstance(o, str):
                raw = raw or o
        rows.append({
            "institution": preferred or raw,
            "institution_raw": raw,
            "city": spec.get("city"),
            "country": spec.get("country"),
            "full_address": spec.get("full_address"),
        })
    return rows


# ---------------------------------------------------------------------------
# RECORD FLATTENING
# ---------------------------------------------------------------------------

def flatten_record(rec, project_id, grant_reference):
    """Convert one WoS record into a flat outcome row tied to a project."""
    item_title, source_title = get_titles(rec)
    ids = get_identifiers(rec)
    core_tc, all_tc = get_times_cited(rec)
    traditional, extended = get_subject_categories(rec)
    macro, meso, micro = get_citation_topics(rec)
    author_kw, plus_kw = get_keywords(rec)
    agencies, grant_ids, fund_text = get_funding(rec)
    authors, rids, orcids = get_authors(rec)

    doctypes = as_list(dig(rec, "static_data", "summary", "doctypes",
                           "doctype", default=[]))
    doctype_str = ";".join(d if isinstance(d, str) else str(d.get("content", ""))
                           for d in doctypes if d)

    return {
        # Join keys back to the project dataset
        "project_id": project_id,
        "grant_reference": grant_reference,
        # Outcome identifiers
        "wos_uid": rec.get("UID"),
        "doi": ids.get("doi") or ids.get("xref_doi"),
        "issn": ids.get("issn"),
        "eissn": ids.get("eissn"),
        # Bibliographic detail
        "title": item_title,
        "source_title": source_title,
        "pub_year": dig(rec, "static_data", "summary", "pub_info", "pubyear"),
        "pub_month": dig(rec, "static_data", "summary", "pub_info", "pubmonth"),
        "cover_date": dig(rec, "static_data", "summary", "pub_info", "coverdate"),
        "sort_date": dig(rec, "static_data", "summary", "pub_info", "sortdate"),
        "early_access_year": dig(rec, "static_data", "summary", "pub_info",
                                 "early_access_year"),
        "doctype": doctype_str,
        "publisher": dig(rec, "static_data", "summary", "publishers", "publisher",
                         "names", "name", "display_name"),
        "open_access_gold": dig(rec, "static_data", "summary", "pub_info",
                                "journal_oas_gold"),
        # Impact measures (the point of the WoS pull for inputs-vs-outputs work)
        "times_cited_core": core_tc,
        "times_cited_all_db": all_tc,
        "usage_180days": dig(rec, "dynamic_data", "wos_usage", "last180days"),
        "usage_alltime": dig(rec, "dynamic_data", "wos_usage", "alltime"),
        "reference_count": dig(rec, "static_data", "fullrecord_metadata", "refs",
                               "count"),
        # People
        "authors": authors,
        "researcher_ids": rids,
        "orcids": orcids,
        "n_addresses": dig(rec, "static_data", "fullrecord_metadata", "addresses",
                           "count"),
        # Funding (evidence for the match, and multi-grant detection)
        "funding_agencies": agencies,
        "funding_grant_ids": grant_ids,
        "funding_text": fund_text,
        # Text
        "abstract": get_abstract(rec),
        "author_keywords": author_kw,
        "keywords_plus": plus_kw,
        # Supplementary taxonomies, QUARANTINED from the discipline crosswalk
        "wos_categories_traditional": traditional,
        "wos_categories_extended": extended,
        "citation_topic_macro": macro,
        "citation_topic_meso": meso,
        "citation_topic_micro": micro,
        "sdg_categories": get_sdgs(rec),
    }


# ---------------------------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------------------------

def search_grant(session, grant_reference, use_cache=True):
    """Return all WoS records acknowledging a grant reference.

    Paginates through the result set. Uses the SQLite cache so re-runs and
    interrupted runs do not repeat API calls.
    """
    cache_key = f"{SEARCH_PREFIX}{grant_reference}"
    if use_cache:
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

    records = []
    first_record = 1
    total = None

    while True:
        params = {
            "databaseId": DATABASE_ID,
            "usrQuery": f"FG=({grant_reference})",
            "count": PAGE_SIZE,
            "firstRecord": first_record,
        }
        payload = request_with_retries(session, BASE_URL, params)
        time.sleep(REQUEST_DELAY)

        if total is None:
            total = dig(payload, "QueryResult", "RecordsFound", default=0) or 0
        if not total:
            break

        page = dig(payload, "Data", "Records", "records", "REC", default=[])
        page = as_list(page) if page else []
        if not page:
            break
        records.extend(page)

        first_record += PAGE_SIZE
        if first_record > total or len(records) >= total:
            break

    cache_put(cache_key, records)
    return records


def fetch_references(session, wos_uid, use_cache=True):
    """Return cited references for one record.

    NOTE: the references endpoint signature has not been verified against a
    live call in this project. It is exercised only with --references, and
    failures are caught per record so a bad signature cannot abort a run.
    """
    cache_key = f"{REFS_PREFIX}{wos_uid}"
    if use_cache:
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

    refs = []
    first_record = 1
    while True:
        params = {
            "databaseId": DATABASE_ID,
            "uniqueId": wos_uid,
            "count": PAGE_SIZE,
            "firstRecord": first_record,
        }
        payload = request_with_retries(session, REFERENCES_URL, params)
        time.sleep(REQUEST_DELAY)

        page = as_list(payload.get("Data", []))
        if not page:
            break
        refs.extend(page)
        total = payload.get("QueryResult", {}).get("RecordsFound", len(refs))
        first_record += PAGE_SIZE
        if first_record > total:
            break

    cache_put(cache_key, refs)
    return refs


# ---------------------------------------------------------------------------
# INPUT / CHECKPOINT
# ---------------------------------------------------------------------------

def resolve_input(explicit):
    """Locate the project list CSV."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            sys.exit(f"Input file not found: {path}")
        return path
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit(
        "No project list found. Pass one with --input. Looked in:\n  "
        + "\n  ".join(str(c) for c in INPUT_CANDIDATES)
    )


def load_projects(path):
    """Return (in_scope_df, excluded_count) split on grant reference format."""
    df = pd.read_csv(path, dtype=str).fillna("")
    if "grant_reference" not in df.columns:
        sys.exit(f"'grant_reference' column not found in {path}")
    if "project_id" not in df.columns:
        sys.exit(f"'project_id' column not found in {path}")

    df["grant_reference"] = df["grant_reference"].str.strip()
    in_scope = df[df["grant_reference"].str.match(COUNCIL_REF_PATTERN, na=False)].copy()
    return in_scope, len(df) - len(in_scope)


def checkpoint_paths(tag):
    """Return (rows_path, done_path) for checkpoint files."""
    return (CKPT_DIR / f"wos_{tag}_rows.json",
            CKPT_DIR / f"wos_{tag}_done.json")


def load_checkpoint(tag):
    rows_path, done_path = checkpoint_paths(tag)
    rows, done = [], set()
    if rows_path.exists():
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if done_path.exists():
        done = set(json.loads(done_path.read_text(encoding="utf-8")))
    return rows, done


def save_checkpoint(tag, rows, done):
    rows_path, done_path = checkpoint_paths(tag)
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect WoS publication outcomes for UKRI GtR projects.")
    parser.add_argument("--input", default=None,
                        help="Project list CSV (defaults to the cleaned dataset).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N in-scope projects (testing).")
    parser.add_argument("--references", action="store_true",
                        help="Also collect cited references for each outcome.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint and start over.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the SQLite response cache.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save progress every N projects (default 25).")
    parser.add_argument("--save-raw", action="store_true",
                        help="Write the raw JSON records per project to data/raw/wos/.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("WOS_API_KEY")
    if not api_key:
        sys.exit("WOS_API_KEY not found. Add it to a local .env file "
                 "(see README > Setup > API keys).")

    input_path = resolve_input(args.input)
    projects, excluded = load_projects(input_path)

    if args.limit:
        projects = projects.head(args.limit)

    tag = "run"
    rows, done = ([], set()) if args.fresh else load_checkpoint(tag)

    print("=" * 74)
    print("WoS Expanded API collection")
    print(f"Input:            {input_path}")
    print(f"In-scope projects: {len(projects)} (council-format grant references)")
    print(f"Excluded:          {excluded} (non-council references, not indexed by WoS)")
    if done:
        print(f"Resuming:          {len(done)} projects already done")
    print(f"References:        {'yes' if args.references else 'no'}")
    print("=" * 74)

    session = requests.Session()
    session.headers.update({"X-ApiKey": api_key, "Accept": "application/json"})

    institution_rows = []
    reference_rows = []
    failures = []

    todo = [r for _, r in projects.iterrows()
            if r["grant_reference"] not in done]

    iterator = tqdm(todo, desc="Projects", unit="proj") if _HAS_TQDM else todo

    for i, row in enumerate(iterator, start=1):
        ref = row["grant_reference"]
        project_id = row["project_id"]

        try:
            records = search_grant(session, ref, use_cache=not args.no_cache)
        except Exception as exc:  # keep going; record the failure
            failures.append({"project_id": project_id, "grant_reference": ref,
                             "error": str(exc)})
            done.add(ref)
            continue

        if args.save_raw and records:
            raw_path = RAW_DIR / f"{ref.replace('/', '_')}.json"
            raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        for rec in records:
            flat = flatten_record(rec, project_id, ref)
            rows.append(flat)

            for inst in get_institutions(rec):
                institution_rows.append({
                    "project_id": project_id,
                    "grant_reference": ref,
                    "wos_uid": rec.get("UID"),
                    **inst,
                })

            if args.references and rec.get("UID"):
                try:
                    for r in fetch_references(session, rec["UID"],
                                              use_cache=not args.no_cache):
                        reference_rows.append({
                            "citing_wos_uid": rec.get("UID"),
                            "project_id": project_id,
                            "grant_reference": ref,
                            "cited_uid": r.get("UID"),
                            "cited_doi": r.get("DOI"),
                            "cited_title": r.get("CitedTitle"),
                            "cited_author": r.get("CitedAuthor"),
                            "cited_work": r.get("CitedWork"),
                            "cited_year": r.get("Year"),
                        })
                except Exception as exc:
                    failures.append({"project_id": project_id,
                                     "grant_reference": ref,
                                     "error": f"references: {exc}"})

        done.add(ref)

        if not _HAS_TQDM and i % 10 == 0:
            print(f"  processed {i}/{len(todo)} projects", flush=True)

        if i % args.checkpoint_every == 0:
            save_checkpoint(tag, rows, done)

    save_checkpoint(tag, rows, done)

    # ----------------------------------------------------------------- output
    if not rows:
        print("\nNo outcomes collected. Nothing written.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    outcomes = pd.DataFrame(rows)
    # Pair-level file: one row per project-paper pair.
    outcomes = outcomes.drop_duplicates(subset=["project_id", "wos_uid"])

    # Paper-level file: deduplicated, so corpus totals are not inflated by
    # papers acknowledging more than one in-scope grant.
    unique = outcomes.drop_duplicates(subset=["wos_uid"]).drop(
        columns=["project_id", "grant_reference"])

    institutions = pd.DataFrame(institution_rows).drop_duplicates()

    def write(df, name):
        latest = PROC_DIR / f"{name}_latest.csv"
        archive = PROC_DIR / f"{name}_{stamp}.csv"
        # A fully resumed run rebuilds outcomes from the checkpoint but collects
        # no fresh institution rows, so this frame can legitimately be empty.
        # Writing it would destroy a good file from a previous run, so refuse.
        if df.empty and latest.exists():
            print(f"    {name}: nothing new collected, keeping the existing file")
            return latest
        df.to_csv(latest, index=False)
        df.to_csv(archive, index=False)
        return latest

    p1 = write(outcomes, "wos_outcomes")
    p2 = write(unique, "wos_outcomes_unique")
    p3 = write(institutions, "wos_outcomes_institutions")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    n_projects_with = outcomes["project_id"].nunique()
    print(f"Projects queried:            {len(done)}")
    print(f"Projects with >=1 outcome:   {n_projects_with}")
    print(f"Project-paper pairs:         {len(outcomes)}")
    print(f"Unique papers:               {len(unique)}")
    print(f"Multi-grant papers:          {len(outcomes) - len(unique)}")
    cited = pd.to_numeric(unique["times_cited_core"], errors="coerce")
    print(f"Times cited populated:       {cited.notna().sum()}/{len(unique)}")
    if cited.notna().any():
        print(f"  median {cited.median():.0f}, max {cited.max():.0f}, "
              f"zero-cited {int((cited == 0).sum())}")
    print(f"Institution rows:            {len(institutions)}")
    if reference_rows:
        refs_df = pd.DataFrame(reference_rows).drop_duplicates()
        p4 = write(refs_df, "wos_outcomes_references")
        print(f"Reference rows:              {len(refs_df)}")
        print(f"  -> {p4}")
    if failures:
        fail_path = PROC_DIR / f"wos_failures_{stamp}.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"Failures:                    {len(failures)} (see {fail_path.name})")

    print(f"\nWritten:\n  {p1}\n  {p2}\n  {p3}")
    print("\nNote: WoS subject categories and Citation Topics are collected but "
          "QUARANTINED from the discipline crosswalk pending a taxonomy decision.")


if __name__ == "__main__":
    main()