"""
harvest_author_identifiers.py
=============================
Recover the author identifiers the collectors fetch and then discard.

Everything here comes from the existing SQLite caches. No API calls are made,
no credentials are needed and nothing is re-collected. The identifiers were
always in the responses; the collectors simply did not keep them.

What was being lost
-------------------
  collect_openalex          kept author.display_name, dropped author.orcid
                            and author.id
  collect_scopus_outcomes   kept ce:indexed-name, dropped @auid and the full
                            ce:given-name
  collect_wos               kept names, researcher_ids and orcids as three
                            INDEPENDENT semicolon-joined lists, so a record
                            with five authors and two ORCIDs gave no way to
                            tell whose ORCID was whose

The last of those is why WoS ORCIDs are currently unusable rather than merely
absent. This script reads each author record whole, so name and identifier
stay attached to each other by construction.

Outputs (data/cleaned/authors/)
-------------------------------
  authors_long.csv
      One row per author position on an outcome. This is the working table.
      Columns: source, outcome_id, outcome_key, doi, author_position,
      raw_name, given_name, surname, name_key, orcid, native_id,
      native_id_type, identity_id, identity_type, project_ids.

      outcome_id is the SAME id the cleaning scripts produce, so this table
      joins directly onto the cleaned outcome files and onto
      data/cleaned/merged/project_outcome_map.csv. doi is the shared
      publication key, which is what lets the same paper be collapsed across
      OpenAlex, Scopus and WoS.

  author_identities.csv
      One row per resolved person. Columns: identity_id, identity_type,
      canonical_name, orcid, n_outcomes, n_sources, n_name_variants,
      name_variants, sources.

  author_harvest_summary.csv
      Recorded figures, one per row, in the same style as the audit.

Identity resolution, in order of trust:
  1. ORCID          crosses all sources, the only genuinely portable key
  2. native id      OpenAlex author id, Scopus AUID, WoS ResearcherID.
                    Resolves duplicates WITHIN a source but not across.
  3. name_key       the string fallback, used only where no identifier exists

Run from the repository root:
    python scripts/enrichment/harvest_author_identifiers.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import re
import sqlite3
import unicodedata

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT_DIR / "cache"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned" / "authors"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LONG_PATH = OUTPUT_DIR / "authors_long.csv"
IDENTITY_PATH = OUTPUT_DIR / "author_identities.csv"
SUMMARY_PATH = FIGURES_DIR / "author_harvest_summary.csv"

RECORDS = []


def record(metric, value, note=""):
    RECORDS.append({"metric": metric, "value": value, "note": note})


# ---------------------------------------------------------------------------
# NAME KEY
# ---------------------------------------------------------------------------
# Deliberately self-contained. This used to import from scripts/diagnostics/,
# which made a pipeline step depend on a diagnostic script that is not tracked,
# so the pipeline would break on any machine without it.
#
# NOTE: this key is lossy by design, reducing a name to surname plus initial.
# It is used ONLY as the last-resort identity when a person has no ORCID and no
# native id, and for the collision analysis below. It is never used to build a
# displayed name, because merging on it conflates different researchers:
# "Zhang, Jianguo" and "Zhang, Jianhui" both reduce to "zhang j".

PARTICLES = {
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "das",
    "du", "la", "le", "el", "al", "bin", "ibn", "ter", "ten", "op", "st",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "prof", "dr"}


def candidate_normalise(name):
    """Reduce a name to "surname initial", lowercased. Comparison key only."""
    if name is None or pd.isna(name):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s,\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    def keep(tokens):
        return [t for t in tokens if t and t.lower() not in SUFFIXES]

    if "," in text:
        surname_part, _, given_part = text.partition(",")
        surname_tokens = keep(surname_part.split())
        given_tokens = keep(given_part.split())
    else:
        tokens = keep(text.split())
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0].lower()
        cut = len(tokens)
        while cut > 1 and len(tokens[cut - 1].replace(".", "")) <= 2:
            cut -= 1
        if cut < len(tokens):
            surname_tokens, given_tokens = tokens[:cut], tokens[cut:]
        else:
            surname_tokens = [tokens[-1]]
            index = len(tokens) - 2
            while index >= 0 and tokens[index].lower() in PARTICLES:
                surname_tokens.insert(0, tokens[index])
                index -= 1
            given_tokens = tokens[: index + 1]

    surname = " ".join(surname_tokens).lower().strip()
    if not surname:
        return ""
    initial = ""
    for token in given_tokens:
        bare = token.replace(".", "")
        if bare:
            initial = bare[0].lower()
            break
    return f"{surname} {initial}".strip()


# ---------------------------------------------------------------------------
# CACHE STREAMING
# ---------------------------------------------------------------------------
# The caches are large (Scopus alone is several hundred MB), so rows are
# streamed rather than loaded into memory all at once.

def stream_cache(db_name, table, key_prefix=None, batch_size=100):
    """Yield (key, parsed_json) from a cache table, skipping unparsable rows."""
    path = CACHE_DIR / db_name
    if not path.exists():
        print(f"  {db_name}: NOT FOUND, skipped")
        return

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if not cursor.fetchone():
        print(f"  {db_name}: table {table} missing, skipped")
        connection.close()
        return

    key_column = "work_id" if table == "work_cache" else "query"
    cursor.execute(f"SELECT {key_column}, response FROM {table}")
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        for key, raw in batch:
            if not raw:
                continue
            if key_prefix and not str(key).startswith(key_prefix):
                continue
            try:
                yield key, json.loads(raw)
            except (TypeError, ValueError):
                continue
    connection.close()


def clean_orcid(value):
    """Reduce an ORCID to its bare 16-character form, or empty."""
    if not value:
        return ""
    text = str(value).strip()
    match = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])", text)
    return match.group(1).upper() if match else ""


def split_display_name(raw, given_hint=""):
    """
    Return (given_name, surname) from a raw display name.

    given_hint is used when the source supplies the given name separately,
    which Scopus does and which is more reliable than splitting the string.
    """
    text = str(raw or "").strip()
    if not text:
        return "", ""
    if given_hint:
        given = str(given_hint).strip()
        surname = text.replace(given, "").strip(" ,")
        return given, surname or text
    if "," in text:
        surname, _, given = text.partition(",")
        return given.strip(), surname.strip()
    parts = text.split()
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


# ---------------------------------------------------------------------------
# HARVESTERS
# ---------------------------------------------------------------------------

def harvest_openalex():
    """
    One row per authorship. work['id'] joins to cleaned openalex_url.

    Two guards, both for OpenAlex data errors observed in this cache:

      1. An ORCID is only trusted when author.id is also present. 518 slots
         (2.5%) have no author id, and 176 of those still carry an ORCID; in
         W2909820344 the null-id authorship for Iakovos Tzanakis carries Bruno
         Lebon's ORCID.
      2. If an ORCID lands on two different authors of the SAME work it is
         dropped for that work, because there is no way to tell which is right.
    """
    print("\nOpenAlex...")
    rows = []
    seen_works = set()
    dropped_no_id = 0
    dropped_duplicate = 0

    for _, payload in stream_cache("openalex_cache.db", "work_cache"):
        works = payload if isinstance(payload, list) else [payload]
        for work in works:
            if not isinstance(work, dict) or "authorships" not in work:
                continue
            work_id = work.get("id") or ""
            if not work_id or work_id in seen_works:
                continue
            seen_works.add(work_id)

            authorships = work.get("authorships") or []

            # Guard 2: find ORCIDs claimed by more than one author here.
            claims = defaultdict(set)
            for authorship in authorships:
                author = (authorship or {}).get("author") or {}
                orcid = clean_orcid(author.get("orcid"))
                if orcid:
                    claims[orcid].add(author.get("id") or "")
            contested = {o for o, ids in claims.items() if len(ids) > 1}

            for position, authorship in enumerate(authorships, 1):
                author = (authorship or {}).get("author") or {}
                if not author:
                    continue
                raw = author.get("display_name") or ""
                given, surname = split_display_name(raw)
                author_id = author.get("id") or ""
                orcid = clean_orcid(author.get("orcid"))

                if orcid and not author_id:
                    orcid = ""
                    dropped_no_id += 1
                elif orcid and orcid in contested:
                    orcid = ""
                    dropped_duplicate += 1

                rows.append({
                    "source": "openalex",
                    "outcome_key": work_id,
                    "author_position": position,
                    "raw_name": raw,
                    "given_name": given,
                    "surname": surname,
                    "name_key": candidate_normalise(raw),
                    "orcid": orcid,
                    "native_id": author_id,
                    "native_id_type": "openalex_author_id",
                })

    print(f"  {len(rows):,} author rows from {len(seen_works):,} works")
    print(f"  ORCIDs dropped, no author id:      {dropped_no_id:,}")
    print(f"  ORCIDs dropped, contested in work: {dropped_duplicate:,}")
    record("openalex_orcid_dropped_no_author_id", dropped_no_id)
    record("openalex_orcid_dropped_contested", dropped_duplicate)
    return rows


def harvest_scopus():
    """
    One row per author. The cache wraps payloads in {"type", "value"} and the
    row key carries the EID, which joins to the cleaned eid column.
    """
    print("\nScopus...")
    rows = []
    seen = set()

    for key, payload in stream_cache(
            "scopus_cache.db", "search_cache", key_prefix="RECORD::"):
        if not isinstance(payload, dict):
            continue
        body = payload.get("value") if "value" in payload else payload
        if not isinstance(body, dict):
            continue
        response = body.get("abstracts-retrieval-response")
        if not isinstance(response, dict):
            continue

        eid = str(key).split("RECORD::", 1)[-1]
        coredata = response.get("coredata") or {}
        eid = coredata.get("eid") or eid
        if eid in seen:
            continue
        seen.add(eid)

        block = response.get("authors")
        if not isinstance(block, dict):
            continue
        authors = block.get("author")
        if isinstance(authors, dict):
            authors = [authors]
        if not isinstance(authors, list):
            continue

        for position, author in enumerate(authors, 1):
            if not isinstance(author, dict):
                continue
            raw = (author.get("ce:indexed-name")
                   or author.get("preferred-name", {}).get("ce:indexed-name")
                   or author.get("ce:surname") or "")
            given = author.get("ce:given-name") or ""
            surname = author.get("ce:surname") or ""
            if not surname:
                given_split, surname = split_display_name(raw, given)
                given = given or given_split
            rows.append({
                "source": "scopus",
                "outcome_key": eid,
                "author_position": int(author.get("@seq") or position),
                "raw_name": raw,
                "given_name": given,
                "surname": surname,
                # Prefer the full given name when Scopus supplies it, since
                # "Smith, John" disambiguates where "Smith J." cannot.
                "name_key": candidate_normalise(
                    f"{surname}, {given}" if given and surname else raw),
                "orcid": clean_orcid(author.get("orcid")),
                "native_id": author.get("@auid") or "",
                "native_id_type": "scopus_auid",
            })

    print(f"  {len(rows):,} author rows from {len(seen):,} records")
    return rows


def normalise_for_match(value):
    """Letters only, lowercased. Used to pair contributors with authors."""
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def wos_contributor_index(static_data):
    """
    Build the AUTHORITATIVE identifier lookup for a WoS record.

    Do NOT trust orcid_id and r_id on static_data.summary.names.name. Those are
    stamped on by sequence position and, measured across this dataset, 43.6% of
    them sit on the wrong author: in WOS:000366070500013 Ji Shouxun's ORCID is
    attached to Yan Feng. 59.7% of records contain at least one such error.

    static_data.contributors.contributor.name carries the full name and the
    identifiers together in one object, so the pairing is explicit rather than
    positional. That is what this reads.

    Returns (by_full_name, by_last_and_initial).
    """
    block = (static_data.get("contributors") or {}).get("contributor")
    if isinstance(block, dict):
        block = [block]
    if not isinstance(block, list):
        return {}, {}

    by_full = {}
    by_initial = defaultdict(list)
    for entry in block:
        name = (entry or {}).get("name") or {}
        if not isinstance(name, dict):
            continue
        last = normalise_for_match(name.get("last_name"))
        first = normalise_for_match(name.get("first_name"))
        if not last:
            continue
        payload = {
            "orcid": clean_orcid(name.get("orcid_id")),
            "r_id": name.get("r_id") or "",
            "first_name": name.get("first_name") or "",
            "last_name": name.get("last_name") or "",
        }
        by_full[(last, first)] = payload
        if first:
            by_initial[(last, first[0])].append(payload)

    return by_full, by_initial


def match_contributor(author, by_full, by_initial):
    """
    Find this author's identifiers, or return blanks.

    Exact normalised surname plus forename first. Falls back to surname plus
    first initial ONLY when that is unambiguous within the record, so a paper
    with two authors called Wang Y never gets a guessed identifier.
    """
    last = normalise_for_match(author.get("last_name"))
    first = normalise_for_match(author.get("first_name"))
    if not last:
        return "", ""

    hit = by_full.get((last, first))
    if hit:
        return hit["orcid"], hit["r_id"]

    if first:
        candidates = by_initial.get((last, first[0]), [])
        if len(candidates) == 1:
            return candidates[0]["orcid"], candidates[0]["r_id"]

    return "", ""


def harvest_wos():
    """
    One row per author, with identifiers taken from the contributors block and
    matched to the author by name.

    The cache holds a bare list of REC objects. REC['UID'] joins to the
    cleaned wos_uid column.
    """
    print("\nWeb of Science...")
    rows = []
    seen = set()
    n_matched = 0
    n_unmatched = 0

    def records_from(payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("Data") or {}
            block = (data.get("Records") or {}).get("records") or {}
            recs = block.get("REC")
            if isinstance(recs, dict):
                return [recs]
            if isinstance(recs, list):
                return recs
            if "static_data" in payload:
                return [payload]
        return []

    for _, payload in stream_cache("wos_cache.db", "search_cache"):
        for rec in records_from(payload):
            if not isinstance(rec, dict):
                continue
            uid = rec.get("UID") or ""
            if not uid or uid in seen:
                continue
            seen.add(uid)

            static_data = rec.get("static_data") or {}
            names = ((static_data.get("summary") or {})
                     .get("names") or {}).get("name")
            if isinstance(names, dict):
                names = [names]
            if not isinstance(names, list):
                continue

            by_full, by_initial = wos_contributor_index(static_data)

            position = 0
            for entry in names:
                if not isinstance(entry, dict):
                    continue
                if entry.get("role") and entry["role"] != "author":
                    continue
                position += 1
                raw = entry.get("display_name") or entry.get("full_name") or ""
                given = entry.get("first_name") or ""
                surname = entry.get("last_name") or ""
                if not surname:
                    given_split, surname = split_display_name(raw)
                    given = given or given_split

                # Identifiers come from the contributors block, matched by
                # name. The orcid_id and r_id sitting on THIS entry are
                # positionally stamped and wrong 43.6% of the time.
                orcid, researcher_id = match_contributor(
                    entry, by_full, by_initial)
                if orcid or researcher_id:
                    n_matched += 1
                else:
                    n_unmatched += 1

                rows.append({
                    "source": "wos",
                    "outcome_key": uid,
                    "author_position": int(entry.get("seq_no") or position),
                    "raw_name": raw,
                    "given_name": given,
                    "surname": surname,
                    "name_key": candidate_normalise(raw),
                    "orcid": orcid,
                    "native_id": researcher_id,
                    "native_id_type": "wos_researcher_id",
                })

    print(f"  {len(rows):,} author rows from {len(seen):,} records")
    print(f"  identifiers matched via contributors block: {n_matched:,}")
    print(f"  no contributor entry for this author:       {n_unmatched:,}")
    record("wos_identifiers_matched", n_matched)
    record("wos_identifiers_unmatched", n_unmatched)
    return rows


# ---------------------------------------------------------------------------
# JOIN BACK TO PROJECTS
# ---------------------------------------------------------------------------

JOIN_SPEC = {
    "openalex": ("openalex_all_outcomes_clean.csv", "openalex_url"),
    "scopus": ("scopus_all_outcomes_clean.csv", "eid"),
    "wos": ("wos_all_outcomes_clean.csv", "outcome_id"),
}


def to_outcome_id(source, outcome_key):
    """
    Convert this script's outcome_key into the outcome_id the cleaning scripts
    use, so the authors table joins directly onto the cleaned outcome files and
    onto project_outcome_map.csv.

    Mirrors exactly what the cleaners do:
      openalex  clean_openalex_outcomes.extract_openalex_id, the last URL segment
      scopus    clean_scopus_outcomes.clean_scopus_outcome_id, minus SCOPUS_ID:,
                reached here by stripping the equivalent "2-s2.0-" EID prefix
      wos       clean_wos_outcomes.clean_wos_outcome_id, minus the WOS: prefix
      gtr       already the GtR outcome id
    """
    key = "" if outcome_key is None or pd.isna(outcome_key) else str(outcome_key).strip()
    if not key:
        return ""
    if source == "openalex":
        return key.rstrip("/").split("/")[-1]
    if source == "scopus":
        return key[7:] if key.lower().startswith("2-s2.0-") else key
    if source == "wos":
        return key[4:] if key.lower().startswith("wos:") else key
    return key


def build_outcome_lookup():
    """Map each source's outcome key to its project ids and its DOI."""
    projects, dois = {}, {}
    for source, (filename, column) in JOIN_SPEC.items():
        path = CLEANED_DIR / filename
        if not path.exists():
            print(f"  {source}: {filename} not found, project ids and DOIs blank")
            continue
        wanted = {column, "project_id", "doi"}
        # dtype=str on the join column specifically - left to its own
        # inference, pandas reads WoS's all-digit outcome_id as an
        # integer and silently drops its leading zero (the raw CSV text
        # itself has it: "001794901400001"), which would otherwise break
        # every WoS lookup below even after the column name is right.
        df = pd.read_csv(path, usecols=lambda c: c in wanted,
                         dtype={column: str}, low_memory=False)
        if column not in df.columns or "project_id" not in df.columns:
            print(f"  {source}: expected columns missing, skipped")
            continue

        grouped = defaultdict(set)
        doi_map = {}
        has_doi = "doi" in df.columns
        for row in df.itertuples(index=False):
            key = getattr(row, column, None)
            project = getattr(row, "project_id", None)
            if pd.isna(key):
                continue
            # Same normalisation as the query side below (to_outcome_id)
            # - both sides need to agree on format. scopus/openalex's
            # cleaned join columns (eid, openalex_url) still carry their
            # source prefix, same as the raw harvested key, so this
            # matches what worked before for them; wos's cleaned
            # outcome_id column already has its prefix stripped, so this
            # is what makes wos match at all.
            key = to_outcome_id(source, key)
            if not pd.isna(project):
                grouped[key].add(str(project))
            if has_doi:
                doi = getattr(row, "doi", None)
                if not pd.isna(doi) and str(doi).strip().lower() not in {"", "nan"}:
                    doi_map.setdefault(key, str(doi).strip().lower())

        projects[source] = grouped
        dois[source] = doi_map
        print(f"  {source}: {len(grouped):,} outcome keys mapped to projects, "
              f"{len(doi_map):,} with a DOI")
    return projects, dois


# ---------------------------------------------------------------------------
# IDENTITY RESOLUTION
# ---------------------------------------------------------------------------

def pick_canonical_name(variants):
    """
    Choose the most informative spelling for a resolved person.

    Prefer a variant carrying a full given name over one with only initials,
    then the longest, then the most frequent. "Helena I. Gomes" beats
    "Gomes HI" because it is the one a reader can actually identify.
    """
    if not variants:
        return ""
    counts = Counter(variants)

    def score(name):
        tokens = re.split(r"[\s,.]+", str(name))
        full_tokens = sum(1 for t in tokens if len(t) > 2)
        return (full_tokens, len(str(name)), counts[name])

    return max(counts, key=score)


def resolve_identities(df):
    """
    Assign every author row an identity, preferring ORCID, then the native id,
    then the name key. Returns the frame with identity columns added.
    """
    identity_ids = []
    identity_types = []

    for orcid, native_id, native_type, name_key in zip(
            df["orcid"], df["native_id"], df["native_id_type"], df["name_key"]):
        if orcid:
            identity_ids.append(f"orcid:{orcid}")
            identity_types.append("orcid")
        elif native_id:
            identity_ids.append(f"{native_type}:{native_id}")
            identity_types.append("native_id")
        else:
            identity_ids.append(f"name:{name_key}")
            identity_types.append("name_only")

    df = df.copy()
    df["identity_id"] = identity_ids
    df["identity_type"] = identity_types
    return df


def main():
    print(f"Repository root: {ROOT_DIR}")
    print(f"Caches:          {CACHE_DIR}")
    print("\nNo API calls are made. Everything below is re-parsed from cache.")

    rows = []
    rows.extend(harvest_openalex())
    rows.extend(harvest_scopus())
    rows.extend(harvest_wos())

    if not rows:
        print("\nNothing harvested. Check the cache/ directory and re-run.")
        return

    df = pd.DataFrame(rows)

    print("\nJoining outcomes to projects and DOIs...")
    project_lookup, doi_lookup = build_outcome_lookup()
    project_ids, doi_values, outcome_ids = [], [], []
    for source, key in zip(df["source"], df["outcome_key"]):
        # Normalise the same way the lookup table's keys were built (e.g.
        # strip WoS's "WOS:" prefix) - using the raw outcome_key here
        # would never match, since project_lookup is keyed by the
        # cleaned/prefix-stripped outcome_id.
        lookup_key = to_outcome_id(source, key)
        projects = project_lookup.get(source, {}).get(lookup_key, set())
        project_ids.append("; ".join(sorted(projects)))
        doi_values.append(doi_lookup.get(source, {}).get(lookup_key, ""))
        outcome_ids.append(lookup_key)
    df["project_ids"] = project_ids
    df["doi"] = doi_values
    df["outcome_id"] = outcome_ids

    matched = sum(1 for v in doi_values if v)
    print(f"  {matched:,} of {len(df):,} author rows carry a DOI "
          f"({100 * matched / len(df):.1f}%)")

    df = resolve_identities(df)

    column_order = [
        "source", "outcome_id", "outcome_key", "doi", "author_position",
        "raw_name", "given_name", "surname", "name_key", "orcid", "native_id",
        "native_id_type", "identity_id", "identity_type", "project_ids",
    ]
    df = df[column_order]
    df.to_csv(LONG_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(df):,} author rows to {LONG_PATH}")

    # ---- Per-source coverage, to reconcile against the audit ----
    print(f"\n{'=' * 74}\nCOVERAGE BY SOURCE\n{'=' * 74}")
    for source, group in df.groupby("source"):
        n = len(group)
        n_orcid = int((group["orcid"] != "").sum())
        n_native = int((group["native_id"] != "").sum())
        n_given = int((group["given_name"].fillna("") != "").sum())
        print(f"\n  {source}")
        print(f"    author rows:      {n:>8,}")
        print(f"    with ORCID:       {n_orcid:>8,}  ({100 * n_orcid / n:5.1f}%)")
        print(f"    with native id:   {n_native:>8,}  ({100 * n_native / n:5.1f}%)")
        print(f"    with given name:  {n_given:>8,}  ({100 * n_given / n:5.1f}%)")
        record(f"{source}_author_rows", n)
        record(f"{source}_with_orcid", n_orcid)
        record(f"{source}_with_native_id", n_native)
        record(f"{source}_with_given_name", n_given)

    # ---- Identity roll-up ----
    identities = []
    for identity_id, group in df.groupby("identity_id"):
        variants = [v for v in group["raw_name"] if str(v).strip()]
        orcids = {o for o in group["orcid"] if o}
        identities.append({
            "identity_id": identity_id,
            "identity_type": group["identity_type"].iloc[0],
            "canonical_name": pick_canonical_name(variants),
            "orcid": "; ".join(sorted(orcids)),
            "n_outcomes": group["outcome_key"].nunique(),
            "n_sources": group["source"].nunique(),
            "sources": "; ".join(sorted(set(group["source"]))),
            "n_name_variants": len(set(variants)),
            "name_variants": "; ".join(sorted(set(variants))),
        })

    identity_df = pd.DataFrame(identities).sort_values(
        ["n_outcomes", "n_name_variants"], ascending=False)
    identity_df.to_csv(IDENTITY_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(identity_df):,} resolved identities to {IDENTITY_PATH}")

    type_counts = identity_df["identity_type"].value_counts()
    print(f"\n{'=' * 74}\nIDENTITY RESOLUTION\n{'=' * 74}")
    for identity_type, count in type_counts.items():
        print(f"  {identity_type:<12} {count:>8,} identities "
              f"({100 * count / len(identity_df):4.1f}%)")
        record(f"identities_{identity_type}", int(count))

    orcid_identities = identity_df[identity_df["identity_type"] == "orcid"]
    if len(orcid_identities):
        multi = orcid_identities[orcid_identities["n_name_variants"] >= 2]
        cross = orcid_identities[orcid_identities["n_sources"] >= 2]
        print(f"\n  ORCID-resolved people:            {len(orcid_identities):,}")
        print(f"  ...written more than one way:     {len(multi):,}")
        print(f"  ...appearing in 2+ sources:       {len(cross):,}")
        print("\n  Those cross-source people are matched on identifier, not")
        print("  on string similarity, so they are correct by construction.")
        record("orcid_identities", len(orcid_identities))
        record("orcid_identities_multi_variant", len(multi))
        record("orcid_identities_cross_source", len(cross))

    # ---- The payoff: identifiers separating people that strings would merge ----
    print(f"\n{'=' * 74}\nWHAT THE IDENTIFIERS FIX\n{'=' * 74}")
    with_orcid = df[df["orcid"] != ""]
    if len(with_orcid):
        per_key = with_orcid.groupby("name_key")["orcid"].nunique()
        collisions = per_key[per_key >= 2]
        print(f"\n  name keys covering more than one real person: "
              f"{len(collisions):,}")
        print(f"  distinct people hidden inside them:            "
              f"{int(collisions.sum()):,}")
        print("\n  These are merges a string-only approach would make silently.")
        print("  ORCID separates them. Worst offenders:")
        for name_key, count in collisions.sort_values(ascending=False).head(8).items():
            print(f"    {name_key:<24} {count:>4} different people")
        record("name_keys_with_multiple_people", len(collisions))
        record("people_hidden_in_colliding_keys", int(collisions.sum()))

        per_orcid = with_orcid.groupby("orcid")["name_key"].nunique()
        split = per_orcid[per_orcid >= 2]
        print(f"\n  people whose name key varies across sources:   {len(split):,}")
        print("  These are matches a string-only approach would MISS.")
        record("people_with_multiple_name_keys", len(split))

    pd.DataFrame(RECORDS).to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"\nFigures written to: {SUMMARY_PATH}\n")


if __name__ == "__main__":
    main()
