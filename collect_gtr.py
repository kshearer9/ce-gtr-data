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

For each kept project the script also enriches with lead organisation name,
principal investigator, funding amount, and dates by following the GtR API
relationship links.

Outputs (in ./data/):
  raw/        -> raw JSON for each search term (kept for reproducibility)
  processed/  -> three CSVs:
    1. gtr_ce_projects_<timestamp>.csv         (kept projects)
    2. gtr_ce_all_with_decision_<timestamp>.csv (all projects + filter decision)
    3. gtr_validation_sample_<timestamp>.csv    (60-project hand-coding sample)

Run example:  python collect_gtr.py --size 25 --max-pages 1
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

_ORG_CACHE = {}
_PERSON_CACHE = {}
_FUND_CACHE = {}


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
    if not href:
        return {}
    try:
        resp = session.get(href, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        time.sleep(delay)
        return resp.json()
    except requests.RequestException:
        return {}


def fetch_fund_value(fund_href, session, delay):
    if not fund_href:
        return ""
    if fund_href in _FUND_CACHE:
        return _FUND_CACHE[fund_href]
    data = fetch_json(fund_href, session, delay)
    vp = data.get("valuePounds")
    value = vp.get("amount", "") if isinstance(vp, dict) else (vp if vp is not None else "")
    _FUND_CACHE[fund_href] = value
    return value


def fetch_org_name(org_href, session, delay):
    if not org_href:
        return ""
    if org_href in _ORG_CACHE:
        return _ORG_CACHE[org_href]
    data = fetch_json(org_href, session, delay)
    name = data.get("name", "")
    _ORG_CACHE[org_href] = name
    return name


def fetch_person_name(person_href, session, delay):
    if not person_href:
        return ""
    if person_href in _PERSON_CACHE:
        return _PERSON_CACHE[person_href]
    data = fetch_json(person_href, session, delay)
    first = data.get("firstName", "") or ""
    other = data.get("otherNames", "") or ""
    surname = data.get("surname", "") or ""
    name = " ".join(p for p in [first, other, surname] if p).strip()
    _PERSON_CACHE[person_href] = name
    return name


# ---------------------------------------------------------------------------
# Per-term collection
# ---------------------------------------------------------------------------
def collect_term(term, size, max_pages, delay, session):
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
# Project flattening
# ---------------------------------------------------------------------------
def flatten_project(project, search_term):
    subjects = project.get("researchSubjects", {}).get("researchSubject", [])
    topics = project.get("researchTopics", {}).get("researchTopic", [])
    
    fund_links = get_links_by_rel(project, "FUND")
    pi_links = get_links_by_rel(project, "PI_PER")
    lead_org_links = get_links_by_rel(project, "LEAD_ORG")
    participant_orgs_links = get_links_by_rel(project, "PARTICIPANT_ORG")

    fund_link = fund_links[0] if fund_links else {}
    pi_href = pi_links[0].get("href", "") if pi_links else ""
    lead_org_href = lead_org_links[0].get("href", "") if lead_org_links else ""
    participant_orgs_hrefs = [link.get("href", "") for link in participant_orgs_links]

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

    lead_funder = project.get("leadFunder", "")
    subjects_with_pct = [s for s in subjects if s.get("percentage", 0) > 0]
    research_subjects_str = format_with_pct(subjects)
    if subjects_with_pct:
        discipline_primary = research_subjects_str
        discipline_source = "research_subjects"
    else:
        discipline_primary = FUNDER_TO_DISCIPLINE.get(lead_funder, lead_funder)
        discipline_source = "funder_mapping"

    return {
        "project_id": project_id,
        "title": title,
        "lead_funder": lead_funder,
        "fund_start": ms_to_month_year(fund_link.get("start")),
        "fund_end": ms_to_month_year(fund_link.get("end")),
        "value_pounds": "",
        "status": project.get("status", ""),
        "grant_category": project.get("grantCategory", ""),
        "grant_reference": grant_ref,
        "principal_investigator": "",
        "lead_organisation": "",
        "participant_organisations": "",
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
        # Store hrefs for enrichment
        "_fund_href": fund_link.get("href", ""),
        "_pi_href": pi_href,
        "_lead_org_href": lead_org_href,
        "_participant_orgs_hrefs": participant_orgs_hrefs,
    }, include


# ---------------------------------------------------------------------------
# Enrichment 
# ---------------------------------------------------------------------------
def enrich_row(row, delay, session):
    """
    Add fund value, PI name, and organisation names.
    Enrichment is done after deduplication and filtering so we only
    make API calls for genuinely CE-relevant projects.
    """
    value_pounds = fetch_fund_value(row.get("_fund_href", ""), session, delay)
    lead_org_name = fetch_org_name(row.get("_lead_org_href", ""), session, delay)
    pi_name = fetch_person_name(row.get("_pi_href", ""), session, delay)
    participant_orgs_names = ""
    participant_hrefs = row.get("_participant_orgs_hrefs", [])
    if participant_hrefs:
        names = [fetch_org_name(href, session, delay) for href in participant_hrefs if href]
        participant_orgs_set = {org.strip() for org in names if org}
        lead_org_name_clean = (lead_org_name or "").strip()
        participant_orgs_set.discard(lead_org_name_clean)
        participant_orgs_names = "; ".join(sorted(participant_orgs_set))
    row["value_pounds"] = value_pounds
    row["lead_organisation"] = lead_org_name
    row["principal_investigator"] = pi_name
    row["participant_organisations"] = participant_orgs_names
    # Remove internal link fields
    for key in ["_fund_href", "_pi_href", "_lead_org_href", "_participant_orgs_hrefs"]:
        row.pop(key, None)
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
    parser.add_argument("--validation-size", type=int, default=60)
    args = parser.parse_args()

    size = max(10, min(args.size, 100))
    terms = (
        [t.strip() for t in args.terms.split(",") if t.strip()]
        if args.terms else DEFAULT_TERMS
    )
    enrich = not args.no_enrich
    apply_filter = not args.no_filter

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
    print(f"Enrich: {enrich} | CE screening: {apply_filter}")

    all_rows = []
    filter_stats = {}
    for term in terms:
        try:
            raw_projects = collect_term(term, size, args.max_pages, args.delay, session)
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
        if apply_filter:
            print(f"    screening: {len(raw_projects)} -> {kept_n} kept ({len(raw_projects) - kept_n} dropped)")

    all_df = pd.DataFrame(all_rows)
    all_df = all_df.drop_duplicates(subset="project_id").reset_index(drop=True)
    if apply_filter:
        kept_df = all_df[all_df["filter_decision"] == "keep"].copy()
    else:
        kept_df = all_df.copy()

    print(f"\n  Unique projects collected: {len(all_df)}")
    print(f"  Unique kept projects: {len(kept_df)}")
    print("\n  Screening summary by term:")
    for term, (raw, kept) in filter_stats.items():
        print(f"    {term:25s}  {raw:>3} raw -> {kept:>3} kept  ({raw - kept} dropped)")
    print("")

    # Enrichment
    if enrich and len(kept_df) > 0:
        total = len(kept_df)
        enriched_rows = []
        for i, (idx, row) in enumerate(kept_df.iterrows(), start=1):
            try:
                enriched = enrich_row(row.to_dict(), args.delay, session)
                enriched_rows.append(enriched)
                print(f"  Enriched {i}/{total} projects.", end="\r")
            except Exception as exc:
                print(f"    Enrichment error (row {idx}): {exc}")
        kept_df = pd.DataFrame(enriched_rows)

    # Output
    out_path = proc_dir / f"gtr_ce_projects_{timestamp}.csv"
    latest_path = proc_dir / "gtr_ce_projects_latest.csv"
    kept_df.to_csv(out_path, index=False, encoding="utf-8")
    kept_df.to_csv(latest_path, index=False, encoding="utf-8")

    all_path = proc_dir / f"gtr_ce_all_with_decision_{timestamp}.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8")

    random.seed(42)
    n = min(args.validation_size, len(all_df))
    sample = all_df.sample(n=n, random_state=42).copy()
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

    print(f"\nOutputs in {proc_dir}/:")
    print(f"  {out_path.name}            (kept projects)")
    print(f"  {all_path.name}  (all projects + screening decision)")
    print(f"  {val_path.name}    (hand-code: fill is_ce_manual with keep/drop)")
    print("\nDone.")


if __name__ == "__main__":
    main()