"""
build_institution_registry.py
=============================
Build a single institution registry that resolves every organisation name in
the dataset to one canonical entity, and re-express the project-organisation
relationship in long form.

Why this step exists
--------------------
Organisation identity is currently carried as free text in two GtR columns
(`lead_organisation`, `participant_organisations`) and as separate affiliation
strings in the Scopus, WoS and OpenAlex outcome tables. The same institution
appears under several surface forms ("The University of Manchester" and
"UNIVERSITY OF MANCHESTER"), which inflates the distinct-institution count and
deflates every concentration measure in RQ1. There is also no organisation-type
field, so universities, research centres and companies sit undifferentiated in
one column.

This script produces the three tables that fix both problems. It deliberately
mirrors the structure already used for `cleaned/authors/author_identities.csv`
(canonical entity + name variants + long-form link table) so the two enrichment
outputs are consistent with each other.

Method
------
1. NORMALISE. Case-fold, strip accents and punctuation, drop the stopwords
   "the", "of", "and". This is what collapses "The University of Manchester"
   onto "University of Manchester". Every merge the normalisation performs is
   written to the report so it can be inspected rather than trusted.

2. TYPE. Assigned from legal-form keywords, in priority order. "Ltd",
   "Limited", "PLC" and "LLP" are legal suffixes rather than descriptive words,
   so this is a deterministic rule and not a judgement call. Organisations that
   match no rule are typed "unknown" and listed for manual review.

3. RESOLVE. Non-company organisations are looked up against the Research
   Organization Registry (ROR), which returns a persistent identifier, a
   display name, an organisation type and a location. ROR is used as the
   authority rather than any of the bibliometric sources because it is
   maintained for exactly this purpose and is independent of whether an
   organisation publishes.

   Companies are NOT sent to ROR. ROR's coverage of small UK companies is
   poor, and the audit found that company names in GtR are already close to
   clean (986 distinct strings collapse to 984 once the legal suffix is
   stripped). They are typed from the suffix, canonicalised within GtR, and
   given a local identifier.

4. TIER THE MATCHES. ROR results are accepted automatically only above a
   pre-declared score threshold. Anything in the middle band is written to a
   review file for manual adjudication and is NOT silently accepted. This
   mirrors the double-coding logic already used for the screening filter.

Outputs (written to data/cleaned/institutions/)
-----------------------------------------------
  institutions.csv               One row per resolved entity.
  institution_name_variants.csv  Every observed surface string, mapped to an
                                 entity, with its source and frequency.
  project_institutions.csv       Long form: one row per project-organisation
                                 pair, carrying the role (lead or participant).
  institution_match_report.txt    Match rates by type and source, plus every
                                 merge the normalisation performed.
  institutions_for_review.csv    Middle-band ROR matches, unknown-type and
                                 ambiguous organisations, for manual
                                 adjudication. Regenerated on every run.

Manual adjudication
-------------------
Fill in `decision_ror` (accept or reject a ROR match), `decision_type` (set
the organisation type) and `decision_note` (why) in
institutions_for_review.csv, then save it in the same folder as
`institution_type_overrides.csv` and re-run. That file is read but never
written by this script, so re-running cannot destroy the coding. Manual
decisions are applied after ROR and beat everything else. Any override whose
institution_id no longer exists is reported rather than silently dropped.

Usage
-----
  # Stage 1, no network needed. Builds everything except the ROR identifiers.
  python3 -m scripts.enrichment.build_institution_registry

  # Stage 2, needs network. Adds ROR ids, canonical names, types, geography.
  python3 -m scripts.enrichment.build_institution_registry --ror

ROR responses are cached in data/cache/ror_lookup.json, so re-running with
--ror does not re-query names that have already been resolved. Deleting that
file forces a fresh lookup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Repo root, or an override for running against a staged copy of the data.
ROOT_DIR = Path(os.environ.get("CE_DATA_ROOT",
                               Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
OUT_DIR = CLEAN_DIR / "institutions"
CACHE_PATH = DATA_DIR / "cache" / "ror_lookup.json"

# Manual adjudications. Written by James, never written by this script, so a
# re-run cannot destroy the coding. See apply_overrides().
OVERRIDES_PATH = CLEAN_DIR / "institutions" / "institution_type_overrides.csv"

PROJECTS_PATH = CLEAN_DIR / "merged" / "projects.csv"
SCOPUS_PATH = CLEAN_DIR / "outcomes" / "scopus_institutions_clean.csv"
WOS_PATH = CLEAN_DIR / "outcomes" / "wos_institutions_clean.csv"
OPENALEX_PATH = CLEAN_DIR / "outcomes" / "openalex_all_outcomes_clean.csv"

# Pre-declared ROR acceptance thresholds. Set before looking at any results.
ROR_AUTO_ACCEPT = 0.90   # accept without review
ROR_REVIEW_FLOOR = 0.70  # below this, treat as no match at all

# ROR returns a LIST of types, unordered, and an organisation can hold several.
# Taking the first is effectively alphabetical: "British Geological Survey" is
# ['funder', 'government'] and would be typed a funder. "Funder" is a role an
# organisation may also have, not its institutional character, so it ranks last
# and is only assigned when nothing else applies. Every organisation appearing
# in this dataset as a lead organisation performed the research; it is not
# there in its capacity as a funder.
ROR_TYPE_PREFERENCE = ["education", "healthcare", "government", "facility",
                       "nonprofit", "company", "archive", "other", "funder"]

ROR_API = "https://api.ror.org/v2/organizations"
ROR_DELAY = 0.4  # seconds between calls, to stay well inside ROR's rate limit

# Organisation-type rules, applied in this order. First match wins.
#
# Ordering is a deliberate choice. Healthcare is tested first so that
# "University Hospital" is not typed as education. Legal-form suffixes (Ltd,
# PLC, LLP) are then tested BEFORE descriptive nouns (centre, institute,
# association), because a legal suffix is a matter of fact registered at
# Companies House whereas "centre" is only a description. The alternative
# ordering types "CITY CENTRE CONTAINERS LIMITED" as a nonprofit.
#
# The cost of this choice is that the UK's research and technology
# organisations, which are mostly companies limited by guarantee, are typed as
# companies: "CENTRE FOR PROCESS INNOVATION LIMITED" and "THE MANUFACTURING
# TECHNOLOGY CENTRE LIMITED" are Catapult centres, not ordinary firms. Rather
# than hard-code a judgement either way, those cases are detected by
# RESEARCH_ORG_WORDS below, sent to ROR for adjudication, and surfaced in the
# review file if ROR does not resolve them.
TYPE_RULES = [
    ("healthcare", r"\bnhs\b|\bhospital\b|\bhealth board\b|\bhealth trust\b"),
    ("education", r"universit|\bcollege\b|\bucl\b|london school of|"
                  r"\blondon sch\b|\bpolytechnic\b|\bschool of\b"),
    ("company", r"\bltd\b|\blimited\b|\bplc\b|\bp l c\b|\bllp\b|\binc\b|"
                r"\bincorporated\b|\bcorp\b|\bcorporation\b|\bgmbh\b|"
                r"\bcompany\b|\bholdings\b|\bpartnership\b|\blp\b|\bbv\b|"
                r"\bab\b|\bcyf\b|\bpty\b"),
    ("government", r"\bcouncil\b|\bborough\b|\bgovernment\b|\bdefra\b|"
                   r"\bministry\b|\bcounty\b|\bagency\b|\bnational health\b"),
    ("nonprofit", r"\binstitute\b|\binstitution\b|\bcentre\b|\bcenter\b|"
                  r"\blaborator|\bacademy\b|\bfoundation\b|\bsociety\b|"
                  r"\bassociation\b|\btrust\b|\bmuseum\b|\bresearch council\b|"
                  r"\bcharity\b|\bcic\b|\bcatapult\b|\bsurvey\b"),
]

# Company-typed organisations whose names suggest a research body rather than
# an ordinary firm. These are sent to ROR even though the suffix rule already
# typed them, so the registry rather than this script decides what they are.
RESEARCH_ORG_WORDS = re.compile(
    r"\binstitute\b|\bcentre\b|\bcenter\b|\blaborator|\bacademy\b|"
    r"\bfoundation\b|\bsociety\b|\bassociation\b|\bcatapult\b|\bcouncil\b|"
    r"\bsurvey\b|\bmuseum\b|\bobservatory\b", re.I)

# Legal suffixes stripped when comparing company names to each other.
#
# "uk" is included, which means "Tata Steel" and "TATA STEEL UK LIMITED" are
# treated as one entity, as are "Syngenta" and "Syngenta UK Limited". This is
# a deliberate choice: for a UK research-ecosystem map the participation is
# what matters, and keeping them apart would create exactly the duplicate
# entities this registry exists to remove. It does conflate a global parent
# with its UK subsidiary, so every entity affected is flagged for review and
# can be split if that conflation is not acceptable.
COMPANY_SUFFIX = re.compile(
    r"\s*\b(ltd|limited|plc|llp|lp|inc|incorporated|gmbh|bv|cic|uk)\b\s*")

# Detects entities that were merged only because "UK" was stripped.
UK_ONLY_MERGE = re.compile(r"\buk\b", re.I)

STOPWORDS = re.compile(r"\b(the|of|and)\b")


# ---------------------------------------------------------------------------
# NORMALISATION AND TYPING
# ---------------------------------------------------------------------------

def normalise(name: str) -> str:
    """Fold an organisation name to a comparison key.

    Case-folds, strips accents to ASCII, replaces punctuation with spaces and
    drops the stopwords "the", "of" and "and". Dropping stopwords is what
    merges "The University of Hull" onto "University of Hull"; it is safe here
    because no two distinct UK institutions in this dataset differ only by a
    stopword. Every merge it causes is listed in the report so this assumption
    is checkable rather than assumed.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode().lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = STOPWORDS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def company_key(name: str) -> str:
    """Normalised key with the legal suffix removed.

    Used only to collapse "CEESOLAR ENERGY LIMITED" onto "CEESOLAR ENERGY LTD".
    Not used for non-companies, where the suffix carries meaning. Full stops
    are removed first so "P.L.C." and "PLC" reduce to the same key.
    """
    return COMPANY_SUFFIX.sub(" ", normalise(re.sub(r"\.", "", str(name)))).strip()


def _match_forms(name) -> list[str]:
    """Two punctuation-normalised forms of a name, for keyword matching."""
    lowered = str(name).lower()
    return [re.sub(r"\s+", " ", re.sub(r"\.", repl, lowered)).strip()
            for repl in ("", " ")]


def classify_type(name: str) -> str:
    """Assign an organisation type from legal-form and descriptive keywords.

    Matching is tried against two punctuation-normalised forms, because the two
    conventions in the data need opposite treatment: "P.L.C." only reduces to
    "plc" if the stops are deleted, while "CO.LIMITED" only reduces to
    "co limited" if they become spaces. Testing both catches each. Mechanical,
    not a judgement.
    """
    forms = _match_forms(name)
    for label, pattern in TYPE_RULES:
        if any(re.search(pattern, form) for form in forms):
            return label
    return "unknown"


def classify_type_from_variants(variants) -> str:
    """Type an entity from every surface form it was observed under.

    Typing the canonical name alone loses information. The canonical name is
    chosen for readability, preferring proper case over block capitals, so
    "RAMBOLL UK LIMITED" and "Ramboll" resolve to "Ramboll", and the legal
    suffix that would have identified it as a company is discarded. Six
    organisations, including Jaguar Land Rover and the Royal Mint, were typed
    "unknown" for this reason alone.

    Rule priority is preserved: each rule is tested against every variant
    before moving to the next rule, so a healthcare match on any variant still
    beats an education match on another.
    """
    forms = [f for v in variants for f in _match_forms(v)]
    for label, pattern in TYPE_RULES:
        if any(re.search(pattern, form) for form in forms):
            return label
    return "unknown"


def needs_ror(entity: dict) -> bool:
    """Should this entity be sent to ROR?

    Everything except ordinary companies. A company-typed entity is still sent
    if its name contains a research-body word, so that Catapult centres and
    similar are adjudicated by the registry rather than by the suffix rule.
    """
    if entity["org_type"] != "company":
        return True
    return bool(RESEARCH_ORG_WORDS.search(entity["canonical_name"]))


def pick_canonical(variants: Counter) -> str:
    """Choose the display form for an entity from its observed variants.

    Preference order: most frequently observed, then the form with the most
    lower-case characters. The second criterion matters because GtR returns
    many organisation names in block capitals, and "University of Sheffield"
    reads better in a figure than "UNIVERSITY OF SHEFFIELD".
    """
    best_count = max(variants.values())
    tied = [v for v, c in variants.items() if c == best_count]
    return sorted(tied, key=lambda s: (-sum(c.islower() for c in s), s))[0]


def slugify(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", key).strip("-")[:60] or "unnamed"


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def split_multi(series: pd.Series) -> list[str]:
    """Explode a semicolon-separated column into a flat list of names."""
    out: list[str] = []
    for value in series.dropna():
        out.extend(p.strip() for p in str(value).split(";") if p.strip())
    return out


def load_sources() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Return the projects frame and a list of (name, source) observations."""
    projects = pd.read_csv(PROJECTS_PATH, low_memory=False)
    observations: list[tuple[str, str]] = []

    for name in projects["lead_organisation"].dropna().astype(str):
        if name.strip():
            observations.append((name.strip(), "gtr"))
    observations += [(n, "gtr")
                     for n in split_multi(projects["participant_organisations"])]

    for path, column, source in [
        (SCOPUS_PATH, "institution", "scopus"),
        (WOS_PATH, "institution", "wos"),
    ]:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            observations += [(str(n).strip(), source)
                             for n in frame[column].dropna()
                             if str(n).strip()]

    if OPENALEX_PATH.exists():
        frame = pd.read_csv(OPENALEX_PATH, low_memory=False)
        observations += [(n, "openalex")
                         for n in split_multi(frame["institutions"])]

    return projects, observations


# ---------------------------------------------------------------------------
# ROR RESOLUTION
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def query_ror(name: str, session) -> dict | None:
    """Query ROR's affiliation matcher and return the best candidate.

    ROR's affiliation endpoint is built for exactly this problem: it takes a
    raw affiliation string and returns ranked organisation candidates with a
    score and a 'chosen' flag. Returns None on any error so a single failed
    lookup cannot abort the run.
    """
    try:
        response = session.get(ROR_API, params={"affiliation": name},
                               timeout=30)
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception as exc:  # noqa: BLE001 - network errors must not abort
        print(f"    ROR lookup failed for {name!r}: {exc}", file=sys.stderr)
        return None
    if not items:
        return None

    best = max(items, key=lambda i: i.get("score", 0))
    org = best.get("organization", {})
    display = next((n["value"] for n in org.get("names", [])
                    if "ror_display" in n.get("types", [])), None)
    location = (org.get("locations") or [{}])[0].get("geonames_details", {})
    return {
        "ror_id": org.get("id"),
        "ror_name": display,
        "ror_types": org.get("types", []),
        "score": best.get("score"),
        "chosen": best.get("chosen"),
        "country_code": location.get("country_code"),
        "city": location.get("name"),
        "lat": location.get("lat"),
        "lon": location.get("lng"),
    }


def resolve_with_ror(entities: dict) -> dict:
    """Look up every non-company entity against ROR, using the disk cache."""
    try:
        import requests
    except ImportError:
        sys.exit("requests is required for --ror. pip install requests")

    cache = load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": "ce-gtr-data institution registry"})

    pending = [key for key, ent in entities.items()
               if needs_ror(ent) and key not in cache]
    print(f"  ROR: {len(cache)} cached, {len(pending)} to look up")

    for index, key in enumerate(pending, 1):
        cache[key] = query_ror(entities[key]["canonical_name"], session)
        if index % 25 == 0:
            print(f"    {index}/{len(pending)}")
            save_cache(cache)
        time.sleep(ROR_DELAY)
    save_cache(cache)
    return cache


def apply_ror(entities: dict, cache: dict) -> None:
    """Fold cached ROR results into the entity records, applying thresholds."""
    for key, entity in entities.items():
        result = cache.get(key)
        if not result or result.get("score") is None:
            entity["match_method"] = "unmatched"
            continue

        score = float(result["score"])
        entity["ror_score"] = round(score, 3)

        # Always record what ROR proposed, even when the match is rejected.
        # Without the proposed name the review file cannot be adjudicated:
        # a score on its own says nothing about whether the match is right.
        entity["ror_matched_name"] = result.get("ror_name")
        entity["ror_country"] = result.get("country_code")

        if score < ROR_REVIEW_FLOOR:
            entity["match_method"] = "unmatched"
            continue

        # The `chosen` flag is ROR's own judgement that a candidate is safe to
        # accept, and it is doing real work: in this dataset a score of 1.00
        # with chosen=false includes "Newcastle University" resolving to the
        # University of Newcastle in Australia, and "Purdue University" to Hue
        # University of Education. Score alone is not sufficient evidence.
        if score < ROR_AUTO_ACCEPT or not result.get("chosen"):
            entity["match_method"] = "ror_review"
            continue  # do not overwrite anything until a human confirms

        # Domain constraint. Every organisation observed in GtR has been
        # funded by UKRI and is therefore overwhelmingly likely to be based in
        # the UK. A non-GB match for such an organisation is suspect: six
        # separate UK local authorities matched to "Council of Southeast
        # Pennsylvania" at auto-accept confidence, and "Royal Mint" matched to
        # the Royal Ottawa Mental Health Centre. Genuine non-UK matches do
        # occur (Neste Oyj, PlasticsEurope) so these are sent to review rather
        # than rejected outright.
        if ("gtr" in entity["sources"]
                and result.get("country_code")
                and result["country_code"] != "GB"):
            entity["match_method"] = "ror_review_non_uk"
            continue

        entity["match_method"] = "ror_auto"

        entity["institution_id"] = "ror:" + str(result["ror_id"]).rsplit("/", 1)[-1]
        entity["id_type"] = "ror"
        entity["ror_id"] = result["ror_id"]
        entity["canonical_name"] = result["ror_name"] or entity["canonical_name"]
        entity["country_code"] = result.get("country_code")
        entity["city"] = result.get("city")
        entity["lat"] = result.get("lat")
        entity["lon"] = result.get("lon")
        types = result.get("ror_types") or []
        if types:
            entity["org_type"] = min(
                types, key=lambda t: (ROR_TYPE_PREFERENCE.index(t)
                                      if t in ROR_TYPE_PREFERENCE
                                      else len(ROR_TYPE_PREFERENCE)))
            entity["type_source"] = "ror"


# ---------------------------------------------------------------------------
# MANUAL ADJUDICATION
# ---------------------------------------------------------------------------

def apply_overrides(entities: dict, cache: dict) -> tuple[int, list[str]]:
    """Apply hand-coded organisation types from the overrides file.

    Some organisations cannot be typed from their name alone. UK Catapult
    centres are the clearest case: "CENTRE FOR PROCESS INNOVATION LIMITED" is
    legally a company limited by guarantee but functions as research
    infrastructure, and which of those it counts as changes what RQ1 says about
    the composition of the ecosystem. That is a judgement about the research
    ecosystem, so it is made by hand and recorded, not inferred by a rule.

    The overrides file is read but never written by this script, so re-running
    the build cannot destroy the coding. Manual decisions are applied last and
    beat both the keyword rules and ROR.

    Expected columns:
        institution_id   must match a row in institutions.csv
        decision_type    education | company | government | nonprofit |
                         healthcare | facility | other. Leave blank to keep
                         the type the rules or ROR assigned.
        decision_ror     accept | reject | blank. "accept" applies a ROR match
                         that was held in the review band, taking its
                         identifier, display name and geography. "reject"
                         records the match as explicitly refused rather than
                         merely unconfirmed. Blank leaves it unconfirmed,
                         which is already the safe default.
        decision_note    free text, the reason. Carried into the registry so
                         the appendix can quote it.

    Returns (number of rows applied, list of ids that no longer exist).
    """
    if not OVERRIDES_PATH.exists():
        return 0, []

    frame = pd.read_csv(OVERRIDES_PATH, dtype=str).fillna("")
    by_id = {e["institution_id"]: e for e in entities.values()}
    key_by_id = {e["institution_id"]: k for k, e in entities.items()}
    applied, stale = 0, []

    for _, row in frame.iterrows():
        target = row.get("institution_id", "").strip()
        want_type = row.get("decision_type", "").strip().lower()
        want_ror = row.get("decision_ror", "").strip().lower()
        if not target or not (want_type or want_ror):
            continue  # a blank row is "not yet coded", not an error
        entity = by_id.get(target)
        if entity is None:
            stale.append(target)
            continue

        if want_ror == "accept":
            result = cache.get(key_by_id.get(target)) or {}
            if result.get("ror_id"):
                entity["institution_id"] = ("ror:" +
                    str(result["ror_id"]).rsplit("/", 1)[-1])
                entity["id_type"] = "ror"
                entity["ror_id"] = result["ror_id"]
                entity["canonical_name"] = (result.get("ror_name")
                                            or entity["canonical_name"])
                entity["country_code"] = result.get("country_code")
                entity["city"] = result.get("city")
                entity["lat"] = result.get("lat")
                entity["lon"] = result.get("lon")
                entity["match_method"] = "ror_manual_accept"
        elif want_ror == "reject":
            # Record the refusal explicitly. A rejected match and a match that
            # was never adjudicated look identical otherwise, and the
            # difference matters when reporting how much of the review pile
            # was actually worked through.
            entity["match_method"] = "ror_manual_reject"
            entity["ror_id"] = None

        if want_type:
            entity["org_type"] = want_type
            entity["type_source"] = "manual"

        entity["decision_note"] = row.get("decision_note", "")
        applied += 1

    return applied, stale


# ---------------------------------------------------------------------------
# CONSOLIDATION
# ---------------------------------------------------------------------------

def consolidate(entities: dict) -> list[dict]:
    """Collapse entities that resolved to the same identifier into one row.

    Entities are grouped by normalised name, so "The Ocean Cleanup" as spelled
    by OpenAlex and by Scopus start as two entities. When both resolve to the
    same ROR record they correctly acquire the same institution_id, but they
    remain two rows. That leaves institutions.csv with a non-unique primary
    key: 590 identifiers were shared by 1,430 rows, and any join to the
    registry fanned out.

    Counting from project_institutions.csv was never affected, because that
    counts identifiers rather than rows. The registry's own row count was.

    Merging is additive: name variants, sources and observation counts are
    unioned or summed, and the strongest provenance wins so a manual decision
    is never overwritten by an automatic one.
    """
    order = {"manual": 3, "ror": 2, "suffix_rule": 1}
    groups: dict[str, list] = defaultdict(list)
    for entity in entities.values():
        groups[entity["institution_id"]].append(entity)

    merged = []
    collapsed = 0
    for institution_id, members in groups.items():
        if len(members) == 1:
            merged.append(members[0])
            continue
        collapsed += len(members) - 1
        # Prefer the record with the strongest type provenance, then the one
        # observed most often, as the base for the merged row.
        base = dict(sorted(members, key=lambda e: (
            -order.get(e["type_source"], 0), -e["n_observations"]))[0])

        variants, sources = set(), set()
        for member in members:
            variants |= set(str(member["name_variants"]).split("; "))
            sources |= {s for s in str(member["sources"]).split("; ") if s}
        base["name_variants"] = "; ".join(sorted(variants))
        base["n_name_variants"] = len(variants)
        base["sources"] = "; ".join(sorted(sources))
        base["n_sources"] = len(sources)
        base["n_observations"] = sum(m["n_observations"] for m in members)
        merged.append(base)

    if collapsed:
        print(f"  consolidated {collapsed} duplicate rows sharing an "
              f"identifier ({len(entities)} -> {len(merged)} institutions)")
    return merged


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build_entities(observations: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Group observed names into entities. Returns (entities, name->key)."""
    # First pass: type each distinct surface form so companies can use the
    # suffix-stripped key while everything else uses the plain normalised key.
    distinct = sorted({name for name, _ in observations})
    key_of: dict[str, str] = {}
    for name in distinct:
        key_of[name] = (company_key(name) if classify_type(name) == "company"
                        else normalise(name))

    variants: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, set] = defaultdict(set)
    for name, source in observations:
        key = key_of[name]
        if not key:
            continue
        variants[key][name] += 1
        sources[key].add(source)

    entities: dict[str, dict] = {}
    for key, counts in variants.items():
        canonical = pick_canonical(counts)
        # Typed from all variants, not just the canonical form. See
        # classify_type_from_variants for why that matters.
        entities[key] = {
            "institution_id": "local:" + slugify(key),
            "id_type": "local",
            "canonical_name": canonical,
            "org_type": classify_type_from_variants(counts),
            "type_source": "suffix_rule",
            "country_code": None, "city": None, "lat": None, "lon": None,
            "ror_id": None, "ror_score": None,
            "ror_matched_name": None, "ror_country": None,
            "match_method": "not_attempted",
            "decision_note": None,
            "sources": "; ".join(sorted(sources[key])),
            "n_sources": len(sources[key]),
            "name_variants": "; ".join(sorted(counts)),
            "n_name_variants": len(counts),
            "n_observations": sum(counts.values()),
        }
    return entities, key_of


def build_project_links(projects: pd.DataFrame, key_of: dict,
                        entities: dict) -> pd.DataFrame:
    """Long-form project-to-institution table with the organisation's role."""
    rows = []
    for _, row in projects.iterrows():
        lead = str(row.get("lead_organisation") or "").strip()
        pairs = [(lead, "lead")] if lead else []
        pairs += [(p.strip(), "participant")
                  for p in str(row.get("participant_organisations") or "").split(";")
                  if p.strip()]
        for name, role in pairs:
            key = key_of.get(name)
            if not key:
                continue
            entity = entities[key]
            rows.append({
                "project_id": row["project_id"],
                "institution_id": entity["institution_id"],
                "institution_name": entity["canonical_name"],
                "org_type": entity["org_type"],
                "name_raw": name,
                "role": role,
                "source": "gtr",
            })
    return pd.DataFrame(rows)


def write_report(entities: dict, key_of: dict, links: pd.DataFrame,
                 ror_used: bool) -> None:
    """Write the match-rate report and the manual-review file."""
    lines = ["INSTITUTION REGISTRY BUILD REPORT", "=" * 60, ""]

    lines.append(f"Distinct surface forms observed: {len(key_of)}")
    lines.append(f"Entity keys before consolidation: {len(entities)}")
    lines.append(f"Distinct institutions:           "
                 f"{len({e['institution_id'] for e in entities.values()})}")
    lines.append("")

    lines.append("Entities by organisation type")
    lines.append("-" * 60)
    counts = Counter(e["org_type"] for e in entities.values())
    for label, count in counts.most_common():
        lines.append(f"  {label:20s} {count:6d}  ({100 * count / len(entities):5.1f}%)")
    lines.append("")

    if ror_used:
        lines.append("ROR match outcome (entities sent to ROR)")
        lines.append("-" * 60)
        lines.append(f"  auto-accept threshold {ROR_AUTO_ACCEPT}, "
                     f"review floor {ROR_REVIEW_FLOOR}")
        eligible = [e for e in entities.values() if needs_ror(e)]
        lines.append("  ror_review = below ROR's own confidence gate; "
                     "ror_review_non_uk = matched outside the UK")
        outcomes = Counter(e["match_method"] for e in eligible)
        for label, count in outcomes.most_common():
            lines.append(f"  {label:20s} {count:6d}  "
                         f"({100 * count / max(len(eligible), 1):5.1f}%)")
        lines.append("")

    lines.append("Normalisation merges (surface forms collapsed onto one entity)")
    lines.append("-" * 60)
    lines.append("Every merge is listed so it can be checked by eye.")
    merged = [(k, e) for k, e in entities.items() if e["n_name_variants"] > 1]
    lines.append(f"  {len(merged)} entities carry more than one surface form.")
    lines.append("")
    for key, entity in sorted(merged, key=lambda kv: -kv[1]["n_observations"]):
        lines.append(f"  {entity['canonical_name']}")
        for variant in sorted(entity["name_variants"].split("; ")):
            lines.append(f"      {variant}")
    lines.append("")

    if not links.empty:
        lines.append("GtR projects by lead-organisation type")
        lines.append("-" * 60)
        lead = links[links["role"] == "lead"]
        for label, count in lead["org_type"].value_counts().items():
            lines.append(f"  {label:20s} {count:6d}  "
                         f"({100 * count / len(lead):5.1f}%)")

    (OUT_DIR / "institution_match_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")

    # Anything a human needs to look at. Restricted to organisations that
    # actually appear in the GtR funding data: the registry also carries
    # several thousand publication-side affiliations, mostly non-UK, which are
    # not in scope for the funding analysis and would swamp the review file.
    review = []
    for entity in entities.values():
        if "gtr" not in entity["sources"]:
            continue
        if (entity["type_source"] == "manual"
                or entity["match_method"] in ("ror_manual_accept",
                                              "ror_manual_reject")):
            continue  # already adjudicated, nothing left to decide
        if entity["match_method"] == "ror_review_non_uk":
            reason = ("ROR matched a non-UK organisation to a UKRI-funded "
                      "body, almost certainly wrong, confirm or reject")
        elif entity["match_method"] == "ror_review":
            reason = "ROR match below the confidence gate, confirm or reject"
        elif entity["org_type"] == "unknown":
            reason = "no type rule matched, assign a type"
        elif (entity["org_type"] == "company"
              and RESEARCH_ORG_WORDS.search(entity["canonical_name"])):
            reason = ("legal form says company but the name suggests a "
                      "research body, confirm the type")
        elif (entity["n_name_variants"] > 1
              and any(UK_ONLY_MERGE.search(v)
                      for v in entity["name_variants"].split("; "))
              and not all(UK_ONLY_MERGE.search(v)
                          for v in entity["name_variants"].split("; "))):
            reason = ("global parent merged with its UK arm because 'UK' is "
                      "stripped, confirm or split")
        else:
            continue
        # A suggested action, so the file can be triaged rather than read row
        # by row. A UKRI-funded organisation resolving to a non-UK institution
        # is wrong often enough that "reject" is the right default; the coder
        # overrides it where the organisation really is foreign-owned.
        country = entity.get("ror_country")
        if country and country != "GB":
            action = "reject the ROR match"
        elif entity["org_type"] == "unknown":
            action = "assign a type"
        else:
            action = "check"
        review.append({**entity, "review_reason": reason,
                       "suggested_action": action})

    if review:
        frame = pd.DataFrame(review)
        # Rows needing thought first, bulk rejections last.
        order = {"check": 0, "assign a type": 1, "reject the ROR match": 2}
        frame["_order"] = frame["suggested_action"].map(order)
        frame = frame.sort_values(["_order", "n_projects"],
                                  ascending=[True, False])
        # Blank decision columns, ready to be filled in and saved as the
        # overrides file. Ordered so the two columns to type into come first.
        frame["decision_type"] = ""
        frame["decision_ror"] = ""
        frame["decision_note"] = ""
        columns = ["institution_id", "canonical_name", "suggested_action",
                   "decision_ror", "decision_type", "decision_note",
                   "ror_matched_name", "ror_country",
                   "ror_score", "org_type", "n_projects", "review_reason",
                   "ror_id", "match_method", "sources", "name_variants"]
        frame[columns].to_csv(OUT_DIR / "institutions_for_review.csv",
                              index=False, encoding="utf-8")
    print(f"  {len(review)} GtR organisations flagged for manual review")
    if review and not OVERRIDES_PATH.exists():
        print(f"  Fill in decision_type, save as {OVERRIDES_PATH.name} in the "
              f"same folder, and re-run to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ror", action="store_true",
                        help="resolve non-company organisations against the "
                             "ROR API (requires network access)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sources...")
    projects, observations = load_sources()
    print(f"  {len(projects)} projects, {len(observations)} name observations")

    print("Grouping into entities...")
    entities, key_of = build_entities(observations)
    print(f"  {len(key_of)} distinct surface forms -> {len(entities)} entities")

    if args.ror:
        print("Resolving against ROR...")
        apply_ror(entities, resolve_with_ror(entities))

    # Manual decisions are applied last so they beat both the keyword rules
    # and ROR.
    applied, stale = apply_overrides(entities, load_cache())
    if applied or stale:
        print(f"Applied {applied} manual type overrides from "
              f"{OVERRIDES_PATH.name}")
    if stale:
        print(f"  WARNING: {len(stale)} override rows reference institution_ids "
              f"that no longer exist and were ignored:")
        for target in stale[:10]:
            print(f"    {target}")
        print("  This usually means the normalisation changed. Re-code these "
              "against the current institutions_for_review.csv.")

    print("Building tables...")
    links = build_project_links(projects, key_of, entities)

    # Project counts fold back onto the entities before anything is written,
    # so the registry and the review file agree.
    per_entity = links.groupby("institution_id")["project_id"].nunique()
    for entity in entities.values():
        entity["n_projects"] = int(per_entity.get(entity["institution_id"], 0))

    registry = pd.DataFrame(consolidate(entities))

    registry["seen_in_gtr"] = registry["sources"].str.contains("gtr")

    columns = ["institution_id", "id_type", "canonical_name", "org_type",
               "type_source", "decision_note", "country_code", "city", "lat",
               "lon", "ror_id", "ror_matched_name", "ror_country",
               "ror_score", "match_method", "seen_in_gtr",
               "n_projects", "n_observations", "n_sources", "sources",
               "n_name_variants", "name_variants"]
    registry = registry[columns].sort_values(
        ["n_projects", "canonical_name"], ascending=[False, True])

    variant_rows = [{"name_raw": name, "name_norm": key,
                     "institution_id": entities[key]["institution_id"]}
                    for name, key in key_of.items() if key in entities]

    registry.to_csv(OUT_DIR / "institutions.csv", index=False, encoding="utf-8")
    pd.DataFrame(variant_rows).to_csv(
        OUT_DIR / "institution_name_variants.csv", index=False, encoding="utf-8")
    links.to_csv(OUT_DIR / "project_institutions.csv", index=False,
                 encoding="utf-8")

    write_report(entities, key_of, links, args.ror)

    print(f"\nWritten to {OUT_DIR}")
    print(f"  institutions.csv              {len(registry)} rows")
    print(f"  institution_name_variants.csv {len(variant_rows)} rows")
    print(f"  project_institutions.csv      {len(links)} rows")
    if not args.ror:
        print("\nROR identifiers not resolved. Re-run with --ror on a machine "
              "with network access to add them.")


if __name__ == "__main__":
    main()
