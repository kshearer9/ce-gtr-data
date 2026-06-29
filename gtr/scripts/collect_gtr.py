"""
collect_gtr.py
==============
Collect UK circular economy research projects from the UKRI Gateway to Research
(GtR) API, screen for genuine CE relevance, and output project-level data
plus a hand-coding sample for validation.

Method: a two-stage protocol following systematic-review conventions (PRISMA
2020; Booth et al. 2016).

  Stage 1 - IDENTIFICATION
    Broad keyword search against the GtR projects API using terms grounded in
    Kirchherr, Reike & Hekkert (2017) and the Ellen MacArthur Foundation
    Circular Economy Glossary (2021).

  Stage 2 - SCREENING
    Each project is classified against a concept-block inclusion rule
    (core / strategy / ambiguous terms with Boolean logic).
    Screening checks title + abstract + technical summary + potential impact.

Flow: all projects are collected and flattened first, deduplicated on
project_id, then screened. Only unique, kept projects are enriched (lead org,
participant orgs, PI, funding value, and - with --sectors - impact sectors from
outcome records), so we never spend API calls on duplicates or dropped projects.

Resuming: enrichment is the slow phase, so for long runs it is checkpointed.
The screened set is saved once, then enriched rows are written to a checkpoint
file every --checkpoint-every projects. If the run is interrupted, simply run
the same command again: it reloads the screened set and the checkpoint, skips
the projects already enriched, and carries on. Delete the checkpoint files (or
use --fresh) to start over.

Outputs (in ./data/):
  raw/        -> raw JSON for each search term (written incrementally, kept for
                 reproducibility)
  processed/  -> three CSVs:
    1. gtr_ce_projects_<timestamp>.csv          (kept projects)
    2. gtr_ce_all_with_decision_<timestamp>.csv (all projects + filter decision)
    3. gtr_validation_sample_<timestamp>.csv    (hand-coding sample)
  checkpoints/ -> resume state (screened set + enriched-so-far); safe to delete
                  once a run has finished cleanly.

Run examples:
    python collect_gtr.py --size 25 --max-pages 1            (quick test)
    python collect_gtr.py --size 25 --max-pages 1 --sectors  (test with sectors)
    python collect_gtr.py --sectors                          (full run + sectors)
    python collect_gtr.py --sectors --fresh                  (ignore any checkpoint)
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
import sqlite3

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


# tqdm gives a clean progress bar (count, %, elapsed, ETA). It is optional:
# if it is not installed, we fall back to a simple "Enriched X/total" counter.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://gtr.ukri.org/gtr/api/projects"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
}

# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

CACHE_DB = CACHE_DIR / "gtr_cache.db"
conn = sqlite3.connect(CACHE_DB)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS api_cache (
    url TEXT PRIMARY KEY,
    response TEXT NOT NULL
)
""")
conn.commit()

CACHE_BUFFER = []
CACHE_BUFFER_SIZE = 200

# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------

def get_cache(url):
    cursor.execute(
        "SELECT response FROM api_cache WHERE url = ?",
        (url,)
    )
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def flush_cache():
    if not CACHE_BUFFER:
        return

    cursor.executemany("""
        INSERT OR REPLACE INTO api_cache (url, response)
        VALUES (?, ?)
    """, CACHE_BUFFER)

    conn.commit()
    CACHE_BUFFER.clear()

def save_cache(url, data):
    CACHE_BUFFER.append((url, json.dumps(data)))

    if len(CACHE_BUFFER) >= CACHE_BUFFER_SIZE:
        flush_cache()

# ---------------------------------------------------------------------------
# Stage 1 - identification: search terms sent to the GtR API
# ---------------------------------------------------------------------------
DEFAULT_TERMS = [
    "circular economy",
    "industrial symbiosis",
    "closed-loop",
    "urban mining",
    "remanufacturing",
    "circular bioeconomy",
]

# ---------------------------------------------------------------------------
# Stage 2 - screening vocabulary (concept blocks)
# ---------------------------------------------------------------------------
# Inclusion rule (in classify_ce):
#   INCLUDE if  (>=1 core)
#           OR  (>=2 strategy)
#           OR  (>=1 ambiguous AND (>=1 strategy OR >=1 core))

CORE_PATTERNS = [
    r"circular econom\w*",
    r"circularity",
    r"industrial symbiosis",
    r"cradle[\s-]to[\s-]cradle",
    r"urban mining",
    r"reverse logistics",
    r"regenerati(?:ve|on)",
    r"technical cycle",
    r"biological cycle",
    r"remanufactur\w*",
    r"upcycl\w*",
    r"downcycl\w*",
    r"design out waste",
    r"circular bioeconom\w*",
    r"bioeconom\w*",
]

STRATEGY_PATTERNS = [
    r"recycl\w*",
    r"reus\w*",
    r"re-us\w*",
    r"refurbish\w*",
    r"repair\w*",
    r"repurpos\w*",
    r"recover\w*",
    r"secondary material\w*",
    r"non-virgin",
    r"product life",
    r"waste hierarchy",
    r"resource efficien\w*",
]

AMBIGUOUS_PATTERNS = [
    r"closed[\s-]loop",
]

# ---------------------------------------------------------------------------
# Funder -> broad discipline mapping
# ---------------------------------------------------------------------------
FUNDER_TO_DISCIPLINE = {
    "EPSRC": "Engineering & Physical Sciences",
    "BBSRC": "Biological Sciences",
    "NERC": "Environmental Sciences",
    "ESRC": "Economic & Social Sciences",
    "AHRC": "Arts & Humanities",
    "MRC": "Medical Research",
    "STFC": "Science & Technology Facilities",
    "Innovate UK": "Industry / Applied",
    "Horizon Europe Guarantee": "International (Horizon Europe)",
    "ISCF": "Industrial Strategy Challenge Fund",
    "SPF": "Strategic Priorities Fund",
    "UKRI FLF": "Future Leaders Fellowship",
    "ISPF": "International Science Partnerships Fund",
    "Ayrton Fund": "International Development (Ayrton)",
    "COVID": "COVID Response",
    "Other NPIF": "National Productivity Investment Fund",
}

# Link rels that point to entities (people, orgs, funding) rather than
# outcomes. Sectors live on outcome records, so we skip these when gathering.
NON_OUTCOME_RELS = {
    "LEAD_ORG", "PI_PER", "FUND", "COLLAB_ORG", "PP_ORG", "FELLOW_PER",
    "CO_INV_PER", "PM_PER", "RESEARCH_PER", "STUDENT_PER", "PARTICIPANT_ORG",
    "KEY_FINDING", "IMPACT_SUMMARY", "COI_PER", "STUDENTSHIP_FROM", 
    "RESEARCH_COI_PER", "TGH_PER", "STUDENT_PP_ORG", "COFUND_ORG",
    "TRANSFER", "TRANSFER_FROM"
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def slugify(text):
    """Turn 'circular economy' into 'circular_economy' for safe filenames."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ms_to_month_year(ms):
    """Convert a GtR millisecond timestamp to 'Mon YYYY' (UTC)."""
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%b %Y")
    except (TypeError, ValueError):
        return ""


def format_with_pct(items):
    """Turn [{text, percentage}, ...] -> 'Energy (50%); ICT (50%)' style string."""
    parts = []
    for item in items:
        text = item.get("text", "").strip()
        pct = item.get("percentage")
        if text and pct is not None:
            parts.append(f"{text} ({pct}%)")
        elif text:
            parts.append(text)
    return "; ".join(parts)


def clean_text(value):
    """Normalise a GtR text field to a single-line string."""
    return (value or "").replace("\n", " ").replace("\r", " ").strip()


def get_grant_ref(identifiers):
    """Extract the primary grant reference (e.g. 'EP/V042432/1') from a
    project's identifiers list. Prefers the RCUK type, falls back to the first
    available. This is the human-readable ID the GtR website uses in URLs."""
    if not identifiers:
        return ""
    for ident in identifiers:
        if ident.get("type") == "RCUK":
            return ident.get("value", "")
    return identifiers[0].get("value", "")


def drop_internal(frame):
    """Remove internal API href columns before exporting."""
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")],
                        errors="ignore")

# Make sure the list-valued href columns are real lists (whether we just
# built them or reloaded them from the checkpoint CSV).
def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v:
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


# ---------------------------------------------------------------------------
# Stage 2 screening
# ---------------------------------------------------------------------------

def find_matches(text, patterns):
    """Return the subset of patterns that match somewhere in text."""
    found = []
    for pat in patterns:
        if re.search(r"\b" + pat, text, re.IGNORECASE):
            found.append(pat)
    return found


def classify_ce(title, abstract, tech_abstract="", potential_impact=""):
    """Apply the concept-block inclusion rule across all available text fields.

    Returns:
        include (bool): True if the project passes Stage 2 screening
        matches (dict): which patterns matched in each block (for audit)
    """
    text = " ".join([
        title or "", abstract or "", tech_abstract or "", potential_impact or "",
    ])
    core = find_matches(text, CORE_PATTERNS)
    strategy = find_matches(text, STRATEGY_PATTERNS)
    ambiguous = find_matches(text, AMBIGUOUS_PATTERNS)

    include = (
        len(core) >= 1
        or len(strategy) >= 2
        or (len(ambiguous) >= 1 and (len(strategy) >= 1 or len(core) >= 1))
    )
    return include, {"core": core, "strategy": strategy, "ambiguous": ambiguous}


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------
# The GtR API can wobble on long runs (read timeouts, transient 502/503/504
# server errors). These are not permanent failures, so rather than giving up we
# retry with exponential backoff. A 404 (page genuinely absent) or 403/4xx is
# NOT retried, since retrying would not help.

# HTTP status codes that are worth retrying (server-side / transient).
RETRYABLE_STATUS = {500, 502, 503, 504, 429}

# Tunable retry behaviour (overridable from the CLI via globals set in main()).
MAX_RETRIES = 5          # attempts per request before giving up
BACKOFF_BASE = 2.0       # seconds; wait grows 2, 4, 8, 16 ... between attempts


def _request_with_retries(session, url, params=None):
    """GET a URL with retry-and-backoff on timeouts and transient server errors.

    Returns the parsed JSON on success. Raises the last exception if every
    attempt fails, so the caller can decide whether to skip or abort.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=60)
            # Retry on transient server statuses; raise on other 4xx/5xx.
            if resp.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(
                    f"{resp.status_code} (retryable) for {resp.url}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            # Only retry timeouts, connection drops, and retryable statuses.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = (
                isinstance(exc, (requests.Timeout, requests.ConnectionError))
                or status in RETRYABLE_STATUS
            )
            last_exc = exc
            if not retryable or attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"\n    (attempt {attempt}/{MAX_RETRIES} failed: {exc}; "
                  f"retrying in {wait:.0f}s)", flush=True)
            time.sleep(wait)
    # Should not reach here, but re-raise defensively.
    raise last_exc


def fetch_page(term, page, size, session, delay):
    """Fetch a single page of GtR project search results for a keyword term."""
    params = {"q": term, "p": page, "s": size}
    data = _request_with_retries(session, BASE_URL, params=params)
    time.sleep(delay)
    return data


def fetch_json(href, session, delay):
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
        data = _request_with_retries(session, href)
        time.sleep(delay)
        # Save to cache
        save_cache(href, data)
        return data

    except requests.RequestException:
        return {}


def fetch_fund_value(fund_href, session, delay):
    """Extract funding amount (GBP) from a fund endpoint."""
    if not fund_href:
        return ""
    data = fetch_json(fund_href, session, delay)
    vp = data.get("valuePounds")
    if isinstance(vp, dict):
        return vp.get("amount", "")
    return vp if vp is not None else ""


def fetch_org_name(org_href, session, delay):
    """Resolve organisation name from GtR organisation endpoint."""
    if not org_href:
        return ""
    data = fetch_json(org_href, session, delay)
    return data.get("name", "")


def fetch_person_name(person_href, session, delay):
    """Resolve PI name from GtR person endpoint."""
    if not person_href:
        return ""
    data = fetch_json(person_href, session, delay)
    first = data.get("firstName", "") or ""
    other = data.get("otherNames", "") or ""
    surname = data.get("surname", "") or ""
    return " ".join(p for p in [first, other, surname] if p).strip()


def fetch_sectors_from_hrefs(hrefs, session, delay):
    """Collect impact sectors from a project's outcome records.
    Sectors are tagged on outcome records (mainly key findings), not on the
    project record itself. Values are captured raw (GtR splits some names on
    internal commas and has occasional source typos); normalisation happens
    later in processing. Returns a semicolon-joined string of unique sectors."""
    if not hrefs:
        return ""
    sectors = []
    seen = set()
    for href in hrefs:
        if not href:
            continue
        data = fetch_json(href, session, delay)
        # safely get sectors block
        sec = data.get("sectors")
        if isinstance(sec, dict):
            for item in sec.get("item", []):
                val = (item or "").strip()
                if val and val not in seen:
                    seen.add(val)
                    sectors.append(val)
    return "; ".join(sectors) if sectors else ""


# ---------------------------------------------------------------------------
# Per-term collection
# ---------------------------------------------------------------------------

def collect_term(term, size, max_pages, session, delay, raw_path):
    """Collect all pages for one search term, writing the raw JSON to disk
    incrementally (one project per line, JSONL) so nothing is held in memory
    longer than needed and a partial run still leaves a usable raw file."""
    print(f"\n  Search term: '{term}'")
    first = fetch_page(term, 1, size, session, delay)
    total_pages = first.get("totalPages", 1)
    total_size = first.get("totalSize", 0)
    print(f"    {total_size} projects across {total_pages} pages")

    if max_pages:
        total_pages = min(total_pages, max_pages)
        print(f"    (limited to {total_pages} page(s) for this run)")

    count = 0
    with open(raw_path, "w", encoding="utf-8") as fh:
        projects = list(first.get("project", []))
        for p in projects:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
            yield p
        count += len(projects)

        for page in range(2, total_pages + 1):
            try:
                data = fetch_page(term, page, size, session, delay)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    break
                raise
            page_projects = data.get("project", [])
            for p in page_projects:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                yield p
            count += len(page_projects)
            print(f"    page {page}/{total_pages} collected", end="\r")
    print(f"    collected {count} raw project records          ")


# ---------------------------------------------------------------------------
# Project flattening (no API calls - just reshapes the search response and
# stashes the hrefs that enrichment will need later)
# ---------------------------------------------------------------------------


def flatten_project(project, search_term):
    """Flatten nested GtR project JSON into a single structured row."""
    subjects = project.get("researchSubjects", {}).get("researchSubject", [])
    topics = project.get("researchTopics", {}).get("researchTopic", [])
    links = project.get("links", {}).get("link", [])

    fund_links, pi_links, lead_org_links, participant_org_links, key_finding_href, outcome_links = [], [], [], [], [], []

    for lk in links:
        href = lk.get("href", "")
        rel = lk.get("rel", "")
        if not href:
            continue
        if rel == "FUND":
            fund_links.append(lk)
        elif rel == "PI_PER":
            pi_links.append(lk)
        elif rel == "LEAD_ORG":
            lead_org_links.append(lk)
        elif rel == "PARTICIPANT_ORG":
            participant_org_links.append(lk)
        elif rel == "KEY_FINDING":
            key_finding_href.append(href)
        elif rel not in NON_OUTCOME_RELS:
            outcome_links.append(lk)

    fund_link = fund_links[0] if fund_links else {}
    fund_href = fund_link.get("href", "")
    pi_href = pi_links[0].get("href", "") if pi_links else ""
    lead_org_href = lead_org_links[0].get("href", "") if lead_org_links else ""
    participant_org_href = [lk.get("href") for lk in participant_org_links]
    outcome_href = [lk.get("href") for lk in outcome_links]

    # Discipline signal (no API call - uses fields already in the response)
    subjects_with_pct = [s for s in subjects if s.get("percentage", 0) > 0]
    research_subjects_str = format_with_pct(subjects)
    lead_funder = project.get("leadFunder", "")
    if subjects_with_pct:
        discipline_primary = research_subjects_str
        discipline_source = "research_subjects"
    else:
        discipline_primary = FUNDER_TO_DISCIPLINE.get(lead_funder, lead_funder)
        discipline_source = "funder_mapping"

    title = project.get("title", "")
    abstract = clean_text(project.get("abstractText"))
    tech_abstract = clean_text(project.get("techAbstractText"))
    potential_impact = clean_text(project.get("potentialImpact"))

    include, matches = classify_ce(title, abstract, tech_abstract, potential_impact)
    project_id = project.get("id", "")

    identifiers_list = project.get("identifiers", {}).get("identifier", [])
    grant_ref = get_grant_ref(identifiers_list)
    gtr_url = (
        f"https://gtr.ukri.org/projects?ref={quote(grant_ref, safe='')}"
        if grant_ref
        else f"https://gtr.ukri.org/projects?ref={project_id}"
    )

    return {
        "project_id": project_id,
        "title": title,
        # Enrichment fields - filled later by enrich_row (blank for now)
        "lead_organisation": "",
        "participant_organisations": "",
        "principal_investigator": "",
        "value_pounds": "",
        "funding_data_available": "",
        "sectors": "",
        # Fields available directly from the search response
        "lead_funder": lead_funder,
        "fund_start": ms_to_month_year(fund_link.get("start")),
        "fund_end": ms_to_month_year(fund_link.get("end")),
        "status": project.get("status", ""),
        "grant_category": project.get("grantCategory", ""),
        "grant_reference": grant_ref,
        "discipline_primary": discipline_primary,
        "discipline_source": discipline_source,
        "research_subjects": research_subjects_str,
        "research_topics": format_with_pct(topics),
        "n_research_subjects": len(subjects_with_pct),
        "abstract_text": abstract,
        "tech_abstract_text": tech_abstract,
        "potential_impact": potential_impact,
        "gtr_url": gtr_url,
        "matched_search_term": search_term,
        "filter_decision": "keep" if include else "drop",
        "core_matches": "; ".join(matches["core"]),
        "strategy_matches": "; ".join(matches["strategy"]),
        "ambiguous_matches": "; ".join(matches["ambiguous"]),
        # Internal href fields (stripped before output)
        "_fund_href": fund_href,
        "_pi_href": pi_href,
        "_lead_org_href": lead_org_href,
        "_participant_org_href": participant_org_href,
        "_outcome_href": outcome_href,
        "_key_finding_href": key_finding_href
    }, include



# ---------------------------------------------------------------------------
# Enrichment (runs after deduplication + screening, so only on unique keepers)
# ---------------------------------------------------------------------------

def enrich_row(row, delay, session, collect_sectors):
    """Add fund value, PI name, organisation names, and (optionally) sectors.
    Enrichment runs after deduplication and filtering so we only make API calls
    for genuinely CE-relevant, unique projects."""
    lead_org_name = fetch_org_name(row.get("_lead_org_href", ""), session, delay)

    # Funding value, plus a flag marking whether GtR actually holds a value
    # (studentships and some other categories carry no per-project funding).
    fund_value = fetch_fund_value(row.get("_fund_href", ""), session, delay)
    row["value_pounds"] = fund_value
    row["funding_data_available"] = bool(
        fund_value and str(fund_value) not in ("", "0", "0.0"))

    row["principal_investigator"] = fetch_person_name(row.get("_pi_href", ""), session, delay)
    row["lead_organisation"] = lead_org_name

    # Participant organisations, deduplicated against the lead org
    participant_href = row.get("_participant_org_href", []) or []
    names = [fetch_org_name(href, session, delay) for href in participant_href if href]
    participant_set = {org.strip() for org in names if org and org.strip()}
    participant_set.discard((lead_org_name or "").strip())
    row["participant_organisations"] = "; ".join(sorted(participant_set))

    # Impact sectors (opt-in via --sectors; follows outcome records)
    if collect_sectors:
        row["sectors"] = fetch_sectors_from_hrefs(
            row.get("_key_finding_href", []) or [], session, delay)
    return row


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume support for the slow enrichment phase)
# ---------------------------------------------------------------------------

def load_checkpoint(enriched_path):
    """Return (list of already-enriched row dicts, set of their project_ids).
    Reads the JSONL checkpoint if it exists, else returns empties."""
    rows, done = [], set()
    if enriched_path.exists():
        with open(enriched_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a half-written final line
                rows.append(rec)
                done.add(str(rec.get("project_id", "")))
    return rows, done


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect CE projects from the UKRI GtR API.")
    parser.add_argument("--terms", type=str, default=None)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default 0.3; raise to be gentler)")
    parser.add_argument("--out-dir", type=str, default=None),
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip fund/org/person lookups (faster, less data)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the CE screening (keep all matches)")
    parser.add_argument("--sectors", action="store_true",
                        help="Also collect impact sectors from outcomes (slow)")
    parser.add_argument("--outcomes", action="store_true", help="Also collect all outcomes")
    parser.add_argument("--validation-size", type=int, default=60)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save enrichment progress every N projects (default 100)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint and start enrichment over")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Retries per request on timeout/server error (default 5)")
    parser.add_argument("--backoff-base", type=float, default=2.0,
                        help="Base seconds for exponential backoff between retries (default 2)")
    args = parser.parse_args()

    # Wire retry settings into the module-level globals the fetchers use.
    global MAX_RETRIES, BACKOFF_BASE
    MAX_RETRIES = max(1, args.max_retries)
    BACKOFF_BASE = max(0.0, args.backoff_base)

    size = max(10, min(args.size, 100))
    terms = (
        [t.strip() for t in args.terms.split(",") if t.strip()]
        if args.terms else DEFAULT_TERMS
    )
    enrich = not args.no_enrich
    apply_filter = not args.no_filter
    collect_sectors = args.sectors
    collect_outcomes = args.outcomes

    # Checkpoint files are keyed to the run configuration (terms + size +
    # sectors) so resuming only ever continues a matching run, never mixes
    # an enrich-with-sectors run with an enrich-without one.
    run_key = f"{'-'.join(slugify(t) for t in terms)}_s{size}_sec{int(collect_sectors)}"
    screened_ckpt = CKPT_DIR / f"screened_{run_key}.csv"
    enriched_ckpt = CKPT_DIR / f"enriched_{run_key}.jsonl"

    if args.fresh:
        for f in (screened_ckpt, enriched_ckpt):
            if f.exists():
                f.unlink()
        print("Starting fresh: existing checkpoint cleared.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    print("=" * 64)
    print("UKRI Gateway to Research - circular economy project collection")
    print("=" * 64)
    print(f"Terms: {len(terms)} | Page size: {size} | Delay: {args.delay}s")
    print(f"Enrich: {enrich} | CE screening: {apply_filter} | Sectors: {collect_sectors}")
    if not _HAS_TQDM:
        print("(tqdm not installed - using a simple counter; "
              "pip install tqdm for a progress bar)")

    # -----------------------------------------------------------------------
    # Phase 1: collect + screen. If a screened checkpoint already exists for
    # this exact run, reload it and skip the searching entirely.
    # -----------------------------------------------------------------------
    if screened_ckpt.exists() and not args.fresh:
        print(f"\nReloading screened set from checkpoint: {screened_ckpt.name}")
        all_df = pd.read_csv(screened_ckpt)
        # Re-derive the href columns is not possible from CSV (they were
        # internal), so the screened checkpoint stores them too - see below.
        print(f"  Unique projects in checkpoint: {len(all_df)}")
    else:
        all_rows = []
        filter_stats = {}
        for term in terms:
            raw_path = RAW_DIR / f"gtr_raw_{slugify(term)}_{timestamp}.jsonl"
            try:
                kept_n = 0
                n_raw = 0
                for p in collect_term(term, size, args.max_pages, session, args.delay, raw_path):
                    row, include = flatten_project(p, term)
                    all_rows.append(row)
                    n_raw += 1
                    if include:
                        kept_n += 1
                filter_stats[term] = (n_raw, kept_n)
            except requests.RequestException as exc:
                # A term failing AFTER all retries means the API is genuinely
                # unavailable. Writing partial output here is the dangerous
                # failure mode (a "complete"-looking file missing whole terms),
                # so we abort loudly and write nothing. Nothing is lost: rerun
                # the same command when the API is back and it starts cleanly.
                print("\n" + "=" * 64, file=sys.stderr)
                print(f"ABORTING: collection failed on term '{term}' after "
                      f"{MAX_RETRIES} retries.", file=sys.stderr)
                print(f"Reason: {exc}", file=sys.stderr)
                print("The GtR API appears to be unavailable or rate-limiting. "
                      "No output\nor checkpoint has been written. Wait a few "
                      "minutes and rerun the\nsame command - it will start "
                      "cleanly.", file=sys.stderr)
                print("=" * 64, file=sys.stderr)
                sys.exit(1)

        if not all_rows:
            print("\nNo projects collected. Check search terms or connection.")
            sys.exit(1)

        all_df = pd.DataFrame(all_rows).drop_duplicates(subset="project_id").reset_index(drop=True)

        print(f"\n  Unique projects collected: {len(all_df)}")
        print("\n  Screening summary by term:")
        for term, (raw, kept) in filter_stats.items():
            print(f"    {term:25s}  {raw:>4} raw -> {kept:>4} kept  ({raw - kept} dropped)")

        # Save the screened checkpoint INCLUDING the internal href columns,
        # so a resumed enrichment run has everything it needs without
        # re-searching. (Lists are JSON-encoded to survive the CSV round-trip.)
        ckpt_df = all_df.copy()
        ckpt_df["_participant_org_href"] = ckpt_df["_participant_org_href"].apply(json.dumps)
        ckpt_df["_key_finding_href"] = ckpt_df["_key_finding_href"].apply(json.dumps)
        ckpt_df.to_csv(screened_ckpt, index=False, encoding="utf-8")
        print(f"\n  Screened set saved to checkpoint: {screened_ckpt.name}")


    if "_participant_org_href" in all_df.columns:
        all_df["_participant_org_href"] = all_df["_participant_org_href"].apply(_as_list)
    if "_key_finding_href" in all_df.columns:
        all_df["_key_finding_href"] = all_df["_key_finding_href"].apply(_as_list)
    if "_outcome_href" in all_df.columns:
        all_df["_outcome_href"] = all_df["_outcome_href"].apply(_as_list)

    if apply_filter:
        kept_df = all_df[all_df["filter_decision"] == "keep"].copy()
    else:
        kept_df = all_df.copy()
    print(f"  Unique kept projects:      {len(kept_df)}")

    # -----------------------------------------------------------------------
    # Phase 2: enrich only the unique keepers, checkpointing as we go.
    # -----------------------------------------------------------------------
    if enrich and len(kept_df) > 0:
        enriched_rows, done_ids = ([], set())
        if not args.fresh:
            enriched_rows, done_ids = load_checkpoint(enriched_ckpt)
            if done_ids:
                print(f"\nResuming enrichment: {len(done_ids)} already done, "
                      f"{len(kept_df) - len(done_ids)} to go")

        todo = kept_df[~kept_df["project_id"].astype(str).isin(done_ids)]
        total = len(kept_df)

        # Open the checkpoint in append mode so each enriched row is persisted
        # as soon as it is produced (JSONL = one JSON object per line).
        ckpt_fh = open(enriched_ckpt, "a", encoding="utf-8")
        try:
            iterator = todo.iterrows()
            if _HAS_TQDM:
                iterator = tqdm(iterator, total=len(todo), initial=0,
                                desc="Enriching", unit="proj")
            processed_since_flush = 0
            for n, (_, row) in enumerate(iterator, start=1):
                try:
                    enriched = enrich_row(row.to_dict(), args.delay, session, collect_sectors)
                    enriched_rows.append(enriched)
                    ckpt_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    processed_since_flush += 1
                except Exception as exc:
                    print(f"\n    Enrichment error on {row.get('project_id','?')}: {exc}")

                if processed_since_flush >= args.checkpoint_every:
                    ckpt_fh.flush()
                    processed_since_flush = 0

                if not _HAS_TQDM:
                    done_now = len(done_ids) + n
                    print(f"  Enriched {done_now}/{total} projects", end="\r")
        finally:
            ckpt_fh.flush()
            ckpt_fh.close()

        kept_df = pd.DataFrame(enriched_rows)
        print(f"\n  Enrichment complete ({len(kept_df)}/{total} projects).")

    # ---- Optional: Collect outcomes (opt-in via --outcomes) ----
    outcome_rows = []
    if collect_outcomes:
        print("\nFetching outcome links...")
        for _, row in kept_df.iterrows():
            for href in row["_outcome_href"]:
                outcome_rows.append({
                    "project_id": row["project_id"],
                    "grant_reference": row["grant_reference"],
                    "title": row["title"],
                    "outcome_href": href,
                })
        outcome_path = RAW_DIR / f"gtr_outcome_hrefs_{timestamp}.csv"
        pd.DataFrame(outcome_rows).to_csv(outcome_path, index=False)
        print(f"Saved {len(outcome_rows)} outcome links to outcome_links.csv")

    kept_df = drop_internal(kept_df)
    all_df_out = drop_internal(all_df)

    # ---- Output 1: kept projects ----
    out_path = PROC_DIR / f"gtr_ce_projects_{timestamp}.csv"
    latest_path = PROC_DIR / "gtr_ce_projects_latest.csv"
    kept_df.to_csv(out_path, index=False, encoding="utf-8")
    kept_df.to_csv(latest_path, index=False, encoding="utf-8")

    # ---- Output 2: full set with filter decisions (audit) ----
    all_path = PROC_DIR / f"gtr_ce_all_with_decision_{timestamp}.csv"
    all_df_out.to_csv(all_path, index=False, encoding="utf-8")

    # ---- Output 3: validation sample for hand-coding ----
    random.seed(42)
    n = min(args.validation_size, len(all_df_out))
    sample = all_df_out.sample(n=n, random_state=42).copy()
    # Coerce to string first: a screened set reloaded from the checkpoint CSV
    # can carry NaN (float) in empty text cells, which would break .str.slice.
    sample["abstract_preview"] = sample["abstract_text"].fillna("").astype(str).str.slice(0, 300)
    sample["tech_abstract_preview"] = sample["tech_abstract_text"].fillna("").astype(str).str.slice(0, 300)
    sample["potential_impact_preview"] = sample["potential_impact"].fillna("").astype(str).str.slice(0, 300)
    sample["is_ce_manual"] = ""
    val_cols = [
        "project_id", "title", "matched_search_term", "filter_decision",
        "core_matches", "strategy_matches", "ambiguous_matches",
        "abstract_preview", "tech_abstract_preview", "potential_impact_preview",
        "gtr_url", "is_ce_manual",
    ]
    val_path = PROC_DIR / f"gtr_validation_sample_{timestamp}.csv"
    sample[val_cols].to_csv(val_path, index=False, encoding="utf-8")

    print(f"\nOutputs in {PROC_DIR}/:")
    print(f"  {out_path.name}            (kept projects)")
    print(f"  {all_path.name}   (all projects + screening decision)")
    print(f"  {val_path.name}      (hand-code: fill is_ce_manual with keep/drop)")
    print(f"\nCheckpoints in {CKPT_DIR}/ (safe to delete now this run finished cleanly).")
    print("\nDone.")

    flush_cache()
    conn.close()


if __name__ == "__main__":
    main()