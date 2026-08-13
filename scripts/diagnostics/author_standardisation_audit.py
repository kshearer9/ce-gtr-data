"""
author_standardisation_audit.py
===============================
Measure the author standardisation problem before deciding how to fix it.

Why this script exists
----------------------
`utils.cleaning.normalise_name` only rewrites names that split into exactly two
whitespace-separated parts. Anything with a middle name, an initial, a particle
(van, de, ter) or a hyphenated surname is returned unchanged. Because the four
sources present names in different surface forms - WoS as "Smith, John A",
OpenAlex as "John Smith" - the same researcher can end up with several distinct
keys and never match across sources.

Separately, the identifiers that would make string matching unnecessary are
being fetched and then discarded:

  - collect_wos.get_authors      builds names, researcher_ids and orcids as
                                 three INDEPENDENT semicolon-joined lists, so
                                 they are not positionally aligned.
  - collect_openalex             keeps author.display_name only, dropping
                                 author.orcid and author.id.
  - collect_scopus_outcomes      keeps ce:indexed-name only, dropping @auid
                                 and the full ce:given-name.

All three collectors cache raw API responses in SQLite, so recovering those
identifiers is a re-parse rather than a re-collection. This script measures
both halves of the picture so the fix can be costed rather than guessed:

  PART A  how much author-string fragmentation exists in the cleaned files,
          and how much of it a stricter normaliser would resolve
  PART B  what ORCID and author-id coverage is sitting unused in the caches
  PART C  the headline go/no-go numbers

Nothing is modified. This script only reads.

Inputs (paths relative to repo root; run from there):
  - data/cleaned/outcomes/{gtr,openalex,scopus,wos}_all_outcomes_clean.csv
  - cache/{openalex_cache.db,scopus_cache.db,wos_cache.db}

Outputs:
  - printed report
  - figures/author_audit_summary.csv   (every quoted number, one per row)

Run:
    python scripts/diagnostics/author_standardisation_audit.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import re
import sqlite3
import sys
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS - three parents up from scripts/diagnostics/, matching the rest of the
# repo. (Note: scripts/cleaning/project_outcome_mapping.py uses four, which is
# a separate issue.)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "author_audit_summary.csv"

# Source file, and the author columns to try in order of preference.
SOURCES = {
    "gtr": {
        "file": "gtr_all_outcomes_clean.csv",
        "raw_cols": ["author", "authors"],
        "clean_cols": ["author_clean", "authors_clean"],
    },
    "openalex": {
        "file": "openalex_all_outcomes_clean.csv",
        "raw_cols": ["authors", "author"],
        "clean_cols": ["authors_clean", "author_clean"],
    },
    "scopus": {
        "file": "scopus_all_outcomes_clean.csv",
        "raw_cols": ["authors", "author"],
        "clean_cols": ["authors_clean", "author_clean"],
    },
    "wos": {
        "file": "wos_all_outcomes_clean.csv",
        "raw_cols": ["authors", "author"],
        "clean_cols": ["authors_clean", "author_clean"],
    },
}

# Recorded values, appended throughout, written out at the end.
RECORDS = []


def record(part, source, metric, value, note=""):
    """Record a number so every figure quoted in the report is traceable."""
    RECORDS.append({
        "part": part,
        "source": source,
        "metric": metric,
        "value": value,
        "note": note,
    })


# ---------------------------------------------------------------------------
# THE CURRENT NORMALISER
# ---------------------------------------------------------------------------
# Import the real one so the numbers describe the repo as it stands. If the
# script is run outside the package context, fall back to a verbatim copy.

try:
    sys.path.insert(0, str(ROOT_DIR))
    from utils.cleaning import normalise_name as current_normalise
    CURRENT_SOURCE = "utils.cleaning.normalise_name (imported)"
except Exception as exc:  # pragma: no cover - depends on run context
    print(f"NOTE: could not import utils.cleaning ({exc}); using inline copy.")
    CURRENT_SOURCE = "inline copy of normalise_name"

    try:
        from nameparser import HumanName
    except ImportError:
        HumanName = None

    def current_normalise(name):
        if pd.isna(name):
            return pd.NA
        name = str(name).strip()
        name = re.sub(r"[,.]", "", name)
        name = " ".join(name.split())
        parts = name.split()
        if len(parts) >= 2 and all(len(p) <= 2 for p in parts[1:]):
            return name
        if len(parts) == 2 and HumanName is not None:
            parsed = HumanName(name)
            if parsed.first and parsed.last:
                return f"{parsed.last} {parsed.first[0]}"
        return name


# ---------------------------------------------------------------------------
# A CANDIDATE NORMALISER
# ---------------------------------------------------------------------------
# Deliberately simple and source-agnostic: reduce every name to
# "surname firstinitial", handling the comma form, particles, hyphenation,
# accents and case. This is NOT proposed as the final answer - it exists so the
# audit can quantify the gap between what the current function achieves and
# what a competent string normaliser would achieve. The ceiling it reveals is
# the point: if it is low, string matching is not the answer and the
# identifier route is the only real fix.

PARTICLES = {
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "das",
    "du", "la", "le", "el", "al", "bin", "ibn", "ter", "ten", "op", "st",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "prof", "dr"}


def strip_accents(text):
    """Fold accented characters to their ASCII base so Muller matches Müller."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenise(text):
    """Split on whitespace, dropping honorifics and generational suffixes."""
    return [
        t for t in text.split()
        if t and t.replace(".", "").lower() not in SUFFIXES
    ]


def is_initial_token(token):
    """
    Is this token an initial rather than a name?

    Length alone is not enough: "Li" in "Wei Li" is a surname while "J" in
    "Smith J" is an initial. Original casing and a trailing full stop are the
    signals that separate them, which is why this runs before lowercasing.
    """
    stripped = token.replace(".", "")
    if not stripped or len(stripped) > 2:
        return False
    return token.endswith(".") or stripped.isupper()


def candidate_normalise(name):
    """
    Reduce a name to "surname firstinitial", lowercase.

    Handles:
      - "Smith, John A"   comma form (WoS, Scopus indexed names)
      - "John Smith"      natural order (OpenAlex, GtR)
      - "Smith J"         surname followed by initials
      - "van der Berg M"  multi-word particles kept with the surname
      - "Smith-Jones, A"  hyphenation preserved
      - accents, case, punctuation, honorifics and suffixes

    Known limitation: names given in an all-capitals natural order where the
    surname is one or two characters, most often CJK names such as "WEI LI",
    are genuinely ambiguous and may be inverted. Title-cased "Wei Li" is
    handled correctly. This is a further argument for keying on identifiers
    rather than strings wherever an identifier exists.
    """
    if pd.isna(name):
        return ""

    # Accents are folded early, but case is preserved until the very end
    # because is_initial_token depends on it.
    text = strip_accents(str(name)).strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s,\-'.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if "," in text:
        # Comma form: everything before the comma is the surname.
        surname_part, _, given_part = text.partition(",")
        surname_tokens = tokenise(surname_part)
        given_tokens = tokenise(given_part)
    else:
        tokens = tokenise(text)
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0].replace(".", "").lower()

        # Trailing initials mean the surname comes first: "Smith J",
        # "van der Berg M", "Smith JA".
        cut = len(tokens)
        while cut > 1 and is_initial_token(tokens[cut - 1]):
            cut -= 1

        if cut < len(tokens):
            surname_tokens = tokens[:cut]
            given_tokens = tokens[cut:]
        else:
            # Natural order: surname is the last token, absorbing any
            # particles that immediately precede it.
            surname_tokens = [tokens[-1]]
            index = len(tokens) - 2
            while index >= 0 and tokens[index].replace(".", "").lower() in PARTICLES:
                surname_tokens.insert(0, tokens[index])
                index -= 1
            given_tokens = tokens[: index + 1]

    surname = " ".join(surname_tokens).replace(".", "").lower().strip()
    if not surname:
        return ""

    initial = ""
    for token in given_tokens:
        stripped = token.replace(".", "")
        if stripped:
            initial = stripped[0].lower()
            break

    return f"{surname} {initial}".strip()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def split_authors(value):
    """Split a delimited author string into individual names."""
    if pd.isna(value):
        return []
    text = str(value)
    # Sources use ";" but "|" appears in older files, so accept both.
    parts = re.split(r"[;|]", text)
    return [p.strip() for p in parts if p and p.strip()]


def pick_column(df, candidates):
    """Return the first candidate column present in the frame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def describe_shape(name):
    """Classify a raw name so the failure mode is visible, not inferred."""
    text = str(name).strip()
    if "," in text:
        return "comma form"
    parts = re.sub(r"[,.]", "", text).split()
    if len(parts) < 2:
        return "single token"
    if all(len(p) <= 2 for p in parts[1:]):
        return "surname + initials"
    if len(parts) == 2:
        return "two parts"
    return "three or more parts"


# ---------------------------------------------------------------------------
# PART A - fragmentation in the cleaned files
# ---------------------------------------------------------------------------

def audit_cleaned_files():
    """Measure author-string fragmentation per source and across sources."""
    print("=" * 78)
    print("PART A - AUTHOR STRING FRAGMENTATION IN THE CLEANED OUTCOMES")
    print("=" * 78)
    print(f"\nCurrent normaliser: {CURRENT_SOURCE}")

    # candidate key -> set of raw strings, and -> set of sources
    key_to_raw = defaultdict(set)
    key_to_sources = defaultdict(set)
    current_to_sources = defaultdict(set)

    any_loaded = False

    for source, config in SOURCES.items():
        path = CLEANED_DIR / config["file"]
        print(f"\n{'-' * 78}\n{source.upper()}  ({config['file']})\n{'-' * 78}")

        if not path.exists():
            print("  FILE NOT FOUND - skipped.")
            record("A", source, "file_found", 0, str(path))
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        except Exception as exc:
            print(f"  COULD NOT READ: {exc}")
            record("A", source, "file_readable", 0, str(exc))
            continue

        any_loaded = True
        record("A", source, "file_found", 1, str(path))
        record("A", source, "n_rows", len(df))

        raw_col = pick_column(df, config["raw_cols"])
        clean_col = pick_column(df, config["clean_cols"])

        print(f"  rows: {len(df):,}")
        print(f"  raw author column:   {raw_col or 'NONE FOUND'}")
        print(f"  clean author column: {clean_col or 'NONE FOUND'}")
        record("A", source, "raw_column", raw_col or "")
        record("A", source, "clean_column", clean_col or "")

        # This is the direct check on whether the original column survives
        # cleaning, which is the thing Kirsty was unsure about.
        record("A", source, "raw_and_clean_both_present",
               int(bool(raw_col) and bool(clean_col)))

        if raw_col is None:
            print("  No author column - skipped.")
            continue

        mentions = []
        for value in df[raw_col]:
            mentions.extend(split_authors(value))

        if not mentions:
            print("  No author mentions parsed - skipped.")
            record("A", source, "n_author_mentions", 0)
            continue

        distinct_raw = set(mentions)
        current_keys = {current_normalise(n) for n in distinct_raw}
        current_keys = {str(k) for k in current_keys if not pd.isna(k)}
        candidate_keys = {candidate_normalise(n) for n in distinct_raw}
        candidate_keys.discard("")

        rows_with_authors = int(df[raw_col].notna().sum())

        print(f"  outcomes with >=1 author: {rows_with_authors:,}")
        print(f"  author mentions:          {len(mentions):,}")
        print(f"  distinct raw strings:     {len(distinct_raw):,}")
        print(f"  distinct after current:   {len(current_keys):,}")
        print(f"  distinct after candidate: {len(candidate_keys):,}")

        record("A", source, "n_outcomes_with_authors", rows_with_authors)
        record("A", source, "n_author_mentions", len(mentions))
        record("A", source, "n_distinct_raw", len(distinct_raw))
        record("A", source, "n_distinct_current", len(current_keys))
        record("A", source, "n_distinct_candidate", len(candidate_keys))

        if distinct_raw:
            current_reduction = 100 * (1 - len(current_keys) / len(distinct_raw))
            candidate_reduction = 100 * (
                1 - len(candidate_keys) / len(distinct_raw))
            print(f"  collapse by current:      {current_reduction:5.1f}%")
            print(f"  collapse by candidate:    {candidate_reduction:5.1f}%")
            record("A", source, "pct_collapse_current",
                   round(current_reduction, 2))
            record("A", source, "pct_collapse_candidate",
                   round(candidate_reduction, 2))

        # How often does the current function change anything at all?
        unchanged = sum(
            1 for n in distinct_raw
            if str(current_normalise(n)).strip() == str(n).strip()
        )
        pct_unchanged = 100 * unchanged / len(distinct_raw)
        print(f"  left unchanged by current:{pct_unchanged:5.1f}%  "
              f"({unchanged:,} of {len(distinct_raw):,})")
        record("A", source, "pct_unchanged_by_current", round(pct_unchanged, 2))

        # Surface form distribution, so the failure mode is evidenced.
        shapes = Counter(describe_shape(n) for n in distinct_raw)
        print("  surface forms:")
        for shape, count in shapes.most_common():
            pct = 100 * count / len(distinct_raw)
            print(f"    {shape:<24} {count:>8,}  ({pct:4.1f}%)")
            record("A", source, f"shape_{shape.replace(' ', '_')}", count,
                   f"{pct:.1f}%")

        for raw in distinct_raw:
            key = candidate_normalise(raw)
            if key:
                key_to_raw[key].add(raw)
                key_to_sources[key].add(source)
            current_key = str(current_normalise(raw)).strip()
            if current_key and current_key.lower() != "nan":
                current_to_sources[current_key].add(source)

    if not any_loaded:
        print("\nNo cleaned outcome files were readable. "
              "Run this from the repository root.")
        return key_to_raw, key_to_sources, current_to_sources

    # ---- The number that actually matters: cross-source matchability ----
    print(f"\n{'=' * 78}\nCROSS-SOURCE MATCHING\n{'=' * 78}")

    current_multi = sum(1 for s in current_to_sources.values() if len(s) >= 2)
    candidate_multi = sum(1 for s in key_to_sources.values() if len(s) >= 2)

    print(f"\n  distinct author keys, current normaliser:   "
          f"{len(current_to_sources):,}")
    print(f"  ...appearing in 2+ sources:                 {current_multi:,}")
    print(f"\n  distinct author keys, candidate normaliser: {len(key_to_sources):,}")
    print(f"  ...appearing in 2+ sources:                 {candidate_multi:,}")

    record("A", "all", "n_keys_current", len(current_to_sources))
    record("A", "all", "n_keys_multi_source_current", current_multi)
    record("A", "all", "n_keys_candidate", len(key_to_sources))
    record("A", "all", "n_keys_multi_source_candidate", candidate_multi)

    if current_multi:
        uplift = 100 * (candidate_multi / current_multi - 1)
        print(f"\n  uplift in cross-source author matches: {uplift:+.1f}%")
        record("A", "all", "pct_uplift_multi_source", round(uplift, 2))

    # ---- Worst fragmentation: one person, many strings ----
    print(f"\n{'-' * 78}\nMOST FRAGMENTED AUTHORS "
          f"(one candidate key, many raw strings)\n{'-' * 78}")
    worst = sorted(key_to_raw.items(), key=lambda kv: -len(kv[1]))[:15]
    for key, raws in worst:
        if len(raws) < 2:
            continue
        sources = ",".join(sorted(key_to_sources[key]))
        print(f"\n  {key!r}  ({len(raws)} variants, sources: {sources})")
        for raw in sorted(raws)[:6]:
            print(f"      {raw!r}")
        if len(raws) > 6:
            print(f"      ... and {len(raws) - 6} more")

    # ---- The honest counterweight: over-collapsing ----
    # A key absorbing many variants may be one person written many ways, or
    # several different people who happen to share a surname and initial.
    # Reporting this is what keeps the candidate numbers from being oversold.
    heavy = [k for k, raws in key_to_raw.items() if len(raws) >= 5]
    print(f"\n{'-' * 78}\nFALSE-MERGE RISK\n{'-' * 78}")
    print(f"\n  candidate keys absorbing 5+ distinct raw strings: {len(heavy):,}")
    print("  These are the keys where 'surname + initial' may be merging")
    print("  genuinely different researchers. They set the ceiling on what")
    print("  any string-only approach can safely deliver.")
    record("A", "all", "n_keys_absorbing_5plus_variants", len(heavy))

    return key_to_raw, key_to_sources, current_to_sources


# ---------------------------------------------------------------------------
# PART B - what the caches are holding that is currently discarded
# ---------------------------------------------------------------------------

def load_cache_rows(db_path, table, column="response"):
    """Yield parsed JSON payloads from a cache table, tolerating bad rows."""
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # Confirm the table exists before querying it.
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cursor.fetchone():
            conn.close()
            return []
        cursor.execute(f"SELECT {column} FROM {table}")
        payloads = []
        for (raw,) in cursor.fetchall():
            if not raw:
                continue
            try:
                payloads.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        conn.close()
        return payloads
    except sqlite3.Error as exc:
        print(f"  SQLite error on {db_path.name}: {exc}")
        return None


def audit_openalex_cache():
    """Count ORCID and author-id coverage sitting unused in the OpenAlex cache."""
    print(f"\n{'-' * 78}\nOPENALEX  (cache/openalex_cache.db, work_cache)\n{'-' * 78}")
    payloads = load_cache_rows(CACHE_DIR / "openalex_cache.db", "work_cache")

    if payloads is None:
        print("  Cache not found - skipped.")
        record("B", "openalex", "cache_found", 0)
        return
    record("B", "openalex", "cache_found", 1)

    n_works = 0
    n_slots = 0
    n_orcid = 0
    n_author_id = 0
    works_with_any_orcid = 0

    for payload in payloads:
        works = payload if isinstance(payload, list) else [payload]
        for work in works:
            if not isinstance(work, dict) or "authorships" not in work:
                continue
            n_works += 1
            has_orcid = False
            for authorship in work.get("authorships") or []:
                author = (authorship or {}).get("author") or {}
                if not author:
                    continue
                n_slots += 1
                if author.get("orcid"):
                    n_orcid += 1
                    has_orcid = True
                if author.get("id"):
                    n_author_id += 1
            if has_orcid:
                works_with_any_orcid += 1

    print(f"  cached works with authorships: {n_works:,}")
    print(f"  author slots:                  {n_slots:,}")
    if n_slots:
        print(f"  with ORCID:                    {n_orcid:,} "
              f"({100 * n_orcid / n_slots:.1f}%)")
        print(f"  with OpenAlex author id:       {n_author_id:,} "
              f"({100 * n_author_id / n_slots:.1f}%)")
    if n_works:
        print(f"  works with >=1 ORCID:          {works_with_any_orcid:,} "
              f"({100 * works_with_any_orcid / n_works:.1f}%)")

    record("B", "openalex", "n_cached_works", n_works)
    record("B", "openalex", "n_author_slots", n_slots)
    record("B", "openalex", "n_with_orcid", n_orcid)
    record("B", "openalex", "n_with_author_id", n_author_id)
    record("B", "openalex", "n_works_with_any_orcid", works_with_any_orcid)
    if n_slots:
        record("B", "openalex", "pct_slots_with_orcid",
               round(100 * n_orcid / n_slots, 2))


def audit_scopus_cache():
    """
    Count Scopus author id, ORCID and given-name coverage in the cache.

    The cache stores each payload inside a {"type": ..., "value": ...}
    envelope rather than at the top level, so the response has to be unwrapped
    before the abstracts-retrieval-response can be reached.
    """
    print(f"\n{'-' * 78}\nSCOPUS  (cache/scopus_cache.db, search_cache)\n{'-' * 78}")
    payloads = load_cache_rows(CACHE_DIR / "scopus_cache.db", "search_cache")

    if payloads is None:
        print("  Cache not found - skipped.")
        record("B", "scopus", "cache_found", 0)
        return
    record("B", "scopus", "cache_found", 1)

    n_records = 0
    n_slots = 0
    n_auid = 0
    n_orcid = 0
    n_given_name = 0

    for payload in payloads:
        if not isinstance(payload, dict):
            continue

        # Unwrap the {"type", "value"} envelope if it is present.
        body = payload.get("value") if "value" in payload else payload
        if not isinstance(body, dict):
            continue

        response = body.get("abstracts-retrieval-response")
        if not isinstance(response, dict):
            continue
        authors_block = response.get("authors")
        if not isinstance(authors_block, dict):
            continue
        author_list = authors_block.get("author")
        if isinstance(author_list, dict):
            author_list = [author_list]
        if not isinstance(author_list, list):
            continue

        n_records += 1
        for author in author_list:
            if not isinstance(author, dict):
                continue
            n_slots += 1
            if author.get("@auid"):
                n_auid += 1
            if author.get("orcid"):
                n_orcid += 1
            if author.get("ce:given-name"):
                n_given_name += 1

    print(f"  cached records with authors: {n_records:,}")
    print(f"  author slots:                {n_slots:,}")
    if n_slots:
        print(f"  with Scopus author id:       {n_auid:,} "
              f"({100 * n_auid / n_slots:.1f}%)")
        print(f"  with ORCID:                  {n_orcid:,} "
              f"({100 * n_orcid / n_slots:.1f}%)")
        print(f"  with full given name:        {n_given_name:,} "
              f"({100 * n_given_name / n_slots:.1f}%)")
        print("    (the collector keeps ce:indexed-name and discards this,")
        print("     which throws away the given name it could disambiguate on)")

    record("B", "scopus", "n_cached_records", n_records)
    record("B", "scopus", "n_author_slots", n_slots)
    record("B", "scopus", "n_with_auid", n_auid)
    record("B", "scopus", "n_with_orcid", n_orcid)
    record("B", "scopus", "n_with_given_name", n_given_name)
    if n_slots:
        record("B", "scopus", "pct_slots_with_auid",
               round(100 * n_auid / n_slots, 2))


def audit_wos_cache():
    """
    Count WoS identifier coverage AND measure the alignment defect.

    collect_wos.get_authors appends to names, researcher_ids and orcids
    independently, so a record with 5 authors and 2 ORCIDs produces lists of
    length 5 and 2 with no way to recover which ORCID belongs to which author.
    The misalignment count below is the size of that problem.
    """
    print(f"\n{'-' * 78}\nWEB OF SCIENCE  (cache/wos_cache.db, search_cache)\n"
          f"{'-' * 78}")
    payloads = load_cache_rows(CACHE_DIR / "wos_cache.db", "search_cache")

    if payloads is None:
        print("  Cache not found - skipped.")
        record("B", "wos", "cache_found", 0)
        return
    record("B", "wos", "cache_found", 1)

    n_records = 0
    n_slots = 0
    n_rid = 0
    n_orcid = 0
    n_misaligned = 0
    n_multi_author_with_partial_ids = 0

    def records_from(payload):
        """
        The collector unwraps the API envelope before caching, so a cached row
        is usually a bare list of REC objects. Accept the envelope form too, in
        case older rows predate that.
        """
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("Data") or {}
            records_block = (data.get("Records") or {}).get("records") or {}
            recs = records_block.get("REC")
            if isinstance(recs, dict):
                return [recs]
            if isinstance(recs, list):
                return recs
            if "static_data" in payload:
                return [payload]
        return []

    for payload in payloads:
        for rec in records_from(payload):
            if not isinstance(rec, dict):
                continue
            names_block = (
                ((rec.get("static_data") or {}).get("summary") or {})
                .get("names") or {}
            )
            names = names_block.get("name")
            if isinstance(names, dict):
                names = [names]
            if not isinstance(names, list):
                continue

            authors = [
                n for n in names
                if isinstance(n, dict)
                and (not n.get("role") or n.get("role") == "author")
            ]
            if not authors:
                continue

            n_records += 1
            n_slots += len(authors)

            rids = sum(1 for a in authors if a.get("r_id"))
            orcids = sum(1 for a in authors if a.get("orcid_id"))
            n_rid += rids
            n_orcid += orcids

            # The defect: id lists shorter than the name list, with no padding.
            if orcids and orcids != len(authors):
                n_misaligned += 1
            if len(authors) > 1 and 0 < orcids < len(authors):
                n_multi_author_with_partial_ids += 1

    print(f"  cached records with authors: {n_records:,}")
    print(f"  author slots:                {n_slots:,}")
    if n_slots:
        print(f"  with researcher id:          {n_rid:,} "
              f"({100 * n_rid / n_slots:.1f}%)")
        print(f"  with ORCID:                  {n_orcid:,} "
              f"({100 * n_orcid / n_slots:.1f}%)")
    if n_records:
        print(f"\n  records where the ORCID list length does not equal the")
        print(f"  author list length:          {n_misaligned:,} "
              f"({100 * n_misaligned / n_records:.1f}%)")
        print(f"  multi-author records with only SOME ORCIDs present,")
        print(f"  ie. silently unattributable: {n_multi_author_with_partial_ids:,}")

    record("B", "wos", "n_cached_records", n_records)
    record("B", "wos", "n_author_slots", n_slots)
    record("B", "wos", "n_with_researcher_id", n_rid)
    record("B", "wos", "n_with_orcid", n_orcid)
    record("B", "wos", "n_records_misaligned", n_misaligned)
    record("B", "wos", "n_records_partial_ids",
           n_multi_author_with_partial_ids)


def audit_caches():
    """Probe all three caches for discarded identifiers."""
    print(f"\n{'=' * 78}")
    print("PART B - IDENTIFIERS ALREADY IN THE CACHES BUT NOT EXTRACTED")
    print("=" * 78)
    print("\nThese are recoverable by re-parsing. No new API calls are needed.")

    audit_openalex_cache()
    audit_scopus_cache()
    audit_wos_cache()


# ---------------------------------------------------------------------------
# PART C - the decision
# ---------------------------------------------------------------------------

def summarise():
    """Print the two or three numbers the decision actually rests on."""
    print(f"\n{'=' * 78}\nPART C - THE DECISION\n{'=' * 78}")

    values = {(r["source"], r["metric"]): r["value"] for r in RECORDS}

    orcid_slots = 0
    total_slots = 0
    any_id_slots = 0
    # Each source has its own native identifier. ORCID is the only one that
    # crosses sources by itself, but a native id still resolves duplicates
    # WITHIN a source, which is where most of the fragmentation sits.
    native_id_metric = {
        "openalex": "n_with_author_id",
        "scopus": "n_with_auid",
        "wos": "n_with_researcher_id",
    }
    for source, metric in native_id_metric.items():
        orcid_slots += values.get((source, "n_with_orcid"), 0) or 0
        total_slots += values.get((source, "n_author_slots"), 0) or 0
        any_id_slots += values.get((source, metric), 0) or 0

    print("\n1. Is the identifier route viable?")
    if total_slots:
        pct_orcid = 100 * orcid_slots / total_slots
        pct_any = 100 * any_id_slots / total_slots
        print(f"   Author slots across OpenAlex, Scopus and WoS: {total_slots:,}")
        print(f"   carrying ORCID:                 {orcid_slots:,} "
              f"({pct_orcid:.1f}%)")
        print(f"   carrying a native author id:    {any_id_slots:,} "
              f"({pct_any:.1f}%)")
        print("   All of it is currently discarded at collection.")
        record("C", "all", "pct_author_slots_with_orcid", round(pct_orcid, 2))
        record("C", "all", "pct_author_slots_with_any_id", round(pct_any, 2))
        if pct_orcid >= 40:
            print("   -> Strong. Key on ORCID across sources, native ids within.")
        elif pct_orcid >= 15:
            print("   -> Partial. ORCID as the cross-source spine, native ids")
            print("      within each source, strings for the remainder.")
        else:
            print("   -> Weak on ORCID. Native ids still deduplicate within a")
            print("      source, but crossing sources will lean on strings.")
    else:
        print("   No cache data read, so this cannot be answered yet.")
        print("   Check the cache/ directory and re-run.")

    print("\n2. How much is the current normaliser leaving on the table?")
    uplift = values.get(("all", "pct_uplift_multi_source"))
    if uplift is not None:
        print(f"   A stricter string normaliser alone would change the number of")
        print(f"   authors matchable across two or more sources by {uplift:+.1f}%.")
    else:
        print("   Not computed - the cleaned outcome files were not readable.")

    print("\n3. What is the ceiling on string matching?")
    heavy = values.get(("all", "n_keys_absorbing_5plus_variants"))
    if heavy is not None:
        noun = "key absorbs" if heavy == 1 else "keys absorb"
        print(f"   {heavy:,} candidate {noun} five or more distinct raw")
        print("   strings. Some are one person written many ways, some are")
        print("   different people sharing a surname and initial. Without an")
        print("   identifier there is no way to tell them apart, and this is")
        print("   the irreducible error in any string-only approach.")

    print(f"\n{'-' * 78}")
    print(f"All figures written to: {OUTPUT_PATH}")
    print(f"{'-' * 78}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"\nRepository root: {ROOT_DIR}")
    print(f"Cleaned outcomes: {CLEANED_DIR}")
    print(f"Caches: {CACHE_DIR}\n")

    audit_cleaned_files()
    audit_caches()
    summarise()

    pd.DataFrame(RECORDS).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
