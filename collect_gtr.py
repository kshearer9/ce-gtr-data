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

Outputs (in ./data/):
  raw/        -> raw JSON for each search term (kept for reproducibility)
  processed/  -> three CSVs:
    1. gtr_ce_projects_<timestamp>.csv         (kept projects)
    2. gtr_ce_all_with_decision_<timestamp>.csv (all projects + filter decision)
    3. gtr_validation_sample_<timestamp>.csv    (60-project hand-coding sample)

Run examples:
    python collect_gtr.py --size 25 --max-pages 1            (quick test)
    python collect_gtr.py --size 25 --max-pages 1 --sectors  (test with sectors)
    python collect_gtr.py --sectors                          (full run + sectors)
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

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://gtr.ukri.org/gtr/api/projects"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
}

# ---------------------------------------------------------------------------
# Stage 1 - identification: search terms sent to the GtR API
# ---------------------------------------------------------------------------
DEFAULT_TERMS = [
    "circular economy",
    "industrial symbiosis",
    "closed-loop",
    "cradle to cradle",
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
}

_CACHE = {}

publication_rows = []


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

def fetch_page(term, page, size, session, delay):
    params = {"q": term, "p": page, "s": size}
    resp = session.get(BASE_URL, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(delay)
    return resp.json()


def fetch_json(href, session, delay):
    """Generic helper: fetch any GtR API URL and return parsed JSON.
    Returns {} on failure so the rest of the script keeps running."""
    if not href:
        return {}
    if href in _CACHE:
        return _CACHE[href]
    try:
        resp = session.get(href, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        time.sleep(delay)
        data = resp.json()
        _CACHE[href] = data
        return data
    except requests.RequestException:
        return {}


def fetch_fund_value(fund_href, session, delay):
    if not fund_href:
        return ""
    data = fetch_json(fund_href, session, delay)
    vp = data.get("valuePounds")
    if isinstance(vp, dict):
        return vp.get("amount", "")
    return vp if vp is not None else ""


def fetch_org_name(org_href, session, delay):
    if not org_href:
        return ""
    data = fetch_json(org_href, session, delay)
    return data.get("name", "")



def fetch_person_name(person_href, session, delay):
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
    sectors = []
    seen = set()
    for href in hrefs:
        if not href:
            continue
        outcome = fetch_json(href, session, delay)
        sec = outcome.get("sectors")
        if isinstance(sec, dict):
            for item in sec.get("item", []):
                val = (item or "").strip()
                if val and val not in seen:
                    seen.add(val)
                    sectors.append(val)
    return "; ".join(sectors)

def fetch_publications(pub_hrefs, project_info, session, delay):
    rows = []
    for href in pub_hrefs:
        data = fetch_json(href, session, delay)
        if not data:
            continue

        rows.append({
            "project_id": project_info["project_id"],
            "grant_reference": project_info["grant_reference"],
            "project_title": project_info["title"],
            "publication_id": data.get("id", ""),
            "publication_title": data.get("title", ""),
            "author": data.get("author", ""),
            "date_published": ms_to_month_year(data.get("datePublished")),
            "publication_type": data.get("type") or "",
            "journal": data.get("journalTitle", ""),
            "doi": data.get("doi", "")
        })
    return rows


# ---------------------------------------------------------------------------
# Per-term collection
# ---------------------------------------------------------------------------

def collect_term(term, size, max_pages, session, delay):
    print(f"\n  Search term: '{term}'")
    first = fetch_page(term, 1, size, session, delay)
    total_pages = first.get("totalPages", 1)
    total_size = first.get("totalSize", 0)
    print(f"    {total_size} projects across {total_pages} pages")

    if max_pages:
        total_pages = min(total_pages, max_pages)
        print(f"    (limited to {total_pages} page(s) for this run)")

    projects = list(first.get("project", []))
    for page in range(2, total_pages + 1):
        try:
            data = fetch_page(term, page, size, session, delay)
            projects.extend(data.get("project", []))
            print(f"    page {page}/{total_pages} collected", end="\r")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break
            raise
    print(f"    collected {len(projects)} raw project records          ")
    return projects


def get_links_by_rel(project, rel):
    links = project.get("links", {}).get("link", [])
    return [lk for lk in links if lk.get("rel") == rel]


# ---------------------------------------------------------------------------
# Project flattening (no API calls - just reshapes the search response and
# stashes the hrefs that enrichment will need later)
# ---------------------------------------------------------------------------

def flatten_project(project, search_term):
    subjects = project.get("researchSubjects", {}).get("researchSubject", [])
    topics = project.get("researchTopics", {}).get("researchTopic", [])
    links = project.get("links", {}).get("link", [])

    fund_links = get_links_by_rel(project, "FUND")
    pi_links = get_links_by_rel(project, "PI_PER")
    lead_org_links = get_links_by_rel(project, "LEAD_ORG")
    participant_org_links = get_links_by_rel(project, "PARTICIPANT_ORG")
    publication_links = get_links_by_rel(project, "PUBLICATION")

    fund_link = fund_links[0] if fund_links else {}
    fund_href = fund_link.get("href", "")
    pi_href = pi_links[0].get("href", "") if pi_links else ""
    lead_org_href = lead_org_links[0].get("href", "") if lead_org_links else ""
    participant_org_hrefs = [lk.get("href", "") for lk in participant_org_links if lk.get("href")]
    publication_hrefs = [lk.get("href") for lk in publication_links if lk.get("href")]

    # Outcome links (everything that isn't a person/org/fund link) - these are
    # what we follow to collect sectors during enrichment.
    outcome_hrefs = [lk.get("href", "") for lk in links
                     if lk.get("rel") not in NON_OUTCOME_RELS and lk.get("href")]

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
        "_participant_org_hrefs": participant_org_hrefs,
        "_publication_hrefs": publication_hrefs,
        "_outcome_hrefs": outcome_hrefs,
    }, include

# ---- Strip internal href columns before writing anything ----
def drop_internal(frame):
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")],
                        errors="ignore")


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
    participant_hrefs = row.get("_participant_org_hrefs", []) or []
    names = [fetch_org_name(href, session, delay) for href in participant_hrefs if href]
    participant_set = {org.strip() for org in names if org and org.strip()}
    participant_set.discard((lead_org_name or "").strip())
    row["participant_organisations"] = "; ".join(sorted(participant_set))

    # Impact sectors (opt-in via --sectors; follows outcome records)
    if collect_sectors:
        row["sectors"] = fetch_sectors_from_hrefs(
            row.get("_outcome_hrefs", []) or [], session, delay)
    return row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect CE projects from the UKRI GtR API.")
    parser.add_argument("--terms", type=str, default=None)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip fund/org/person lookups (faster, less data)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the CE screening (keep all matches)")
    parser.add_argument("--sectors", action="store_true",
                        help="Also collect impact sectors from outcomes (slow)")
    parser.add_argument("--publications", action="store_true", help="Also collect all publications")
    parser.add_argument("--validation-size", type=int, default=60)
    args = parser.parse_args()

    size = max(10, min(args.size, 100))
    terms = (
        [t.strip() for t in args.terms.split(",") if t.strip()]
        if args.terms else DEFAULT_TERMS
    )
    enrich = not args.no_enrich
    apply_filter = not args.no_filter
    collect_sectors = args.sectors
    collect_publications = args.publications

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    print("=" * 64)
    print("UKRI Gateway to Research - circular economy project collection")
    print("=" * 64)
    print(f"Terms: {len(terms)} | Page size: {size} | Delay: {args.delay}s")
    print(f"Enrich: {enrich} | CE screening: {apply_filter} | Sectors: {collect_sectors}")

    all_rows = []
    filter_stats = {}

    # ---- Collect + flatten everything first (no enrichment yet) ----
    for term in terms:
        try:
            raw_projects = collect_term(term, size, args.max_pages, session, args.delay)
        except requests.RequestException as exc:
            print(f"    ERROR for term '{term}': {exc}", file=sys.stderr)
            continue

        raw_path = raw_dir / f"gtr_raw_{slugify(term)}_{timestamp}.json"
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump(raw_projects, fh, indent=2, ensure_ascii=False)

        kept_n = 0
        for p in raw_projects:
            row, include = flatten_project(p, term)
            all_rows.append(row)
            if include:
                kept_n += 1
        filter_stats[term] = (len(raw_projects), kept_n)

    if not all_rows:
        print("\nNo projects collected. Check search terms or connection.")
        sys.exit(1)

    # ---- Deduplicate on project_id, then screen ----
    all_df = pd.DataFrame(all_rows).drop_duplicates(subset="project_id").reset_index(drop=True)
    if apply_filter:
        kept_df = all_df[all_df["filter_decision"] == "keep"].copy()
    else:
        kept_df = all_df.copy()

    print(f"\n  Unique projects collected: {len(all_df)}")
    print(f"  Unique kept projects:      {len(kept_df)}")
    print("\n  Screening summary by term:")
    for term, (raw, kept) in filter_stats.items():
        print(f"    {term:25s}  {raw:>3} raw -> {kept:>3} kept  ({raw - kept} dropped)")

    # ---- Enrich only the unique keepers ----
    if enrich and len(kept_df) > 0:
        total = len(kept_df)
        enriched_rows = []
        for i, (_, row) in enumerate(kept_df.iterrows(), start=1):
            try:
                enriched_rows.append(enrich_row(row.to_dict(), args.delay, session, collect_sectors))
            except Exception as exc:
                print(f"\n    Enrichment error: {exc}")
            print(f"  Enriched {i}/{total} projects", end="\r")
        kept_df = pd.DataFrame(enriched_rows)
        print(f"\n  Enrichment complete ({total} projects).")

    # ---- Collect publications (optional) ----
    if collect_publications:
        print("\n  Fetching publications...")
        for _, row in kept_df.iterrows():
            try:
                publication_rows.extend(fetch_publications(
                    row["_publication_hrefs"], row, session, args.delay))
            except Exception as e:
                print(f"Publication error: {e}")

    kept_df = drop_internal(kept_df)
    all_df_out = drop_internal(all_df)

    # ---- Output 1: kept projects ----
    out_path = proc_dir / f"gtr_ce_projects_{timestamp}.csv"
    latest_path = proc_dir / "gtr_ce_projects_latest.csv"
    kept_df.to_csv(out_path, index=False, encoding="utf-8")
    kept_df.to_csv(latest_path, index=False, encoding="utf-8")

    # ---- Output 2: full set with filter decisions (audit) ----
    all_path = proc_dir / f"gtr_ce_all_with_decision_{timestamp}.csv"
    all_df_out.to_csv(all_path, index=False, encoding="utf-8")

    # ---- Output 3: validation sample for hand-coding ----
    random.seed(42)
    n = min(args.validation_size, len(all_df_out))
    sample = all_df_out.sample(n=n, random_state=42).copy()
    sample["abstract_preview"] = sample["abstract_text"].str.slice(0, 300)
    sample["tech_abstract_preview"] = sample["tech_abstract_text"].str.slice(0, 300)
    sample["potential_impact_preview"] = sample["potential_impact"].str.slice(0, 300)
    sample["is_ce_manual"] = ""
    val_cols = [
        "project_id", "title", "matched_search_term", "filter_decision",
        "core_matches", "strategy_matches", "ambiguous_matches",
        "abstract_preview", "tech_abstract_preview", "potential_impact_preview",
        "gtr_url", "is_ce_manual",
    ]
    val_path = proc_dir / f"gtr_validation_sample_{timestamp}.csv"
    sample[val_cols].to_csv(val_path, index=False, encoding="utf-8")

    # ---- Output 4: project outputs ----
    publication_df = pd.DataFrame(publication_rows)
    if collect_publications:
        pub_path = proc_dir / f"gtr_outputs_{timestamp}.csv"
        publication_df.to_csv(pub_path, index=False, encoding="utf-8")
        publication_df.to_csv(proc_dir / "gtr_outputs_latest.csv",
                            index=False, encoding="utf-8")

    print(f"\nOutputs in {proc_dir}/:")
    print(f"  {out_path.name}            (kept projects)")
    print(f"  {all_path.name}   (all projects + screening decision)")
    print(f"  {val_path.name}      (hand-code: fill is_ce_manual with keep/drop)")
    if collect_publications:
        print(f"  {pub_path.name}      (all publications)")
    print("\nDone.")


if __name__ == "__main__":
    main()