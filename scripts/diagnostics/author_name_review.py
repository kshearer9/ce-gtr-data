"""
author_name_review.py
=====================
Produce human-reviewable files showing what standardising author names would
actually do, so the merges can be inspected before anything is changed.

This is the companion to author_standardisation_audit.py. The audit counts the
problem; this writes out the names themselves.

Outputs (in figures/):

  author_name_groups.csv
      One row per proposed standardised key. This is the review file. Columns:
        candidate_key           the proposed standardised name
        n_variants              how many distinct raw spellings collapse here
        n_mentions              total author mentions across all sources
        n_sources               how many of the four sources contribute
        sources                 which ones
        n_distinct_given_names  distinct FULL given names in the group
        distinct_given_names    what they are
        review_flag             see below
        n_current_keys          how many keys the CURRENT function leaves these as
        variants                every raw spelling, semicolon separated

  author_name_map.csv
      One row per distinct raw author string, with its source, its mention
      count, what the current normaliser makes of it, and what the candidate
      normaliser makes of it. Use this to look up individual names.

The review_flag is the column that matters:

  LIKELY DIFFERENT PEOPLE   two or more distinct full given names share this
                            surname and initial, eg "Jianguo Zhang" and
                            "Jianhui Zhang" both becoming "zhang j". Merging
                            these would be wrong.
  PROBABLY SAFE             the variants differ only by punctuation, spacing,
                            case or initials, with at most one full given name.
  NO GIVEN NAME             only initials are available anywhere in the group,
                            so it cannot be judged from the strings alone.

Nothing is modified. This script only reads.

Run:
    python scripts/diagnostics/author_name_review.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import importlib.util
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"
OUTPUT_DIR = ROOT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS_PATH = OUTPUT_DIR / "author_name_groups.csv"
MAP_PATH = OUTPUT_DIR / "author_name_map.csv"

SOURCES = {
    "gtr": ("gtr_all_outcomes_clean.csv", ["author", "authors"]),
    "openalex": ("openalex_all_outcomes_clean.csv", ["authors", "author"]),
    "scopus": ("scopus_all_outcomes_clean.csv", ["authors", "author"]),
    "wos": ("wos_all_outcomes_clean.csv", ["authors", "author"]),
}

# ---------------------------------------------------------------------------
# Reuse the audit's normalisers rather than restating them, so the review file
# and the audit numbers can never drift apart.
# ---------------------------------------------------------------------------

AUDIT_PATH = Path(__file__).resolve().parent / "author_standardisation_audit.py"
spec = importlib.util.spec_from_file_location("author_audit", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
sys.modules["author_audit"] = audit
spec.loader.exec_module(audit)

candidate_normalise = audit.candidate_normalise
current_normalise = audit.current_normalise
split_authors = audit.split_authors
pick_column = audit.pick_column
strip_accents = audit.strip_accents
PARTICLES = audit.PARTICLES
SUFFIXES = audit.SUFFIXES


def name_parts_in(raw, candidate_key):
    """
    Return (full_given_names, middle_initials) found in a raw string.

    "Jianguo Zhang"      -> ({"jianguo"}, set())
    "Williams, Paul A."  -> ({"paul"}, {"a"})
    "Zhang, J. G."       -> (set(), {"g"})

    The surname is removed first using the candidate key, so a middle name is
    picked up but the surname never is. Middle initials matter because
    "Paul A. Williams", "Paul N. Williams" and "Paul T. Williams" share a given
    name but are three different researchers, and a surname-plus-initial key
    would silently merge them.
    """
    if not candidate_key:
        return set(), set()
    surname = candidate_key.rsplit(" ", 1)[0] if " " in candidate_key else candidate_key
    surname_tokens = set(surname.split())

    cleaned = strip_accents(str(raw)).lower()
    for character in ",.'":
        cleaned = cleaned.replace(character, " ")
    # Split runs of initials such as "y.c." or "mr" into separate tokens only
    # where they are clearly initials, ie short and following the surname.
    cleaned = cleaned.replace("-", " ")

    tokens = [
        t for t in cleaned.split()
        if t and t not in surname_tokens
        and t not in PARTICLES and t not in SUFFIXES
    ]

    full_names = set()
    short_tokens = []
    for token in tokens:
        if len(token) > 2:
            full_names.add(token)
        else:
            short_tokens.append(token)

    # The first given token is the one the candidate key already uses, so only
    # anything AFTER it counts as a distinguishing middle initial.
    middle_initials = set()
    if full_names:
        # A full given name is present, so every short token is a middle initial.
        for token in short_tokens:
            middle_initials.update(token)
    elif len(short_tokens) >= 1:
        # Initials only: drop the first character, the rest are middle initials.
        flattened = "".join(short_tokens)
        middle_initials.update(flattened[1:])

    return full_names, middle_initials


def main():
    print(f"Repository root: {ROOT_DIR}")
    print(f"Reading: {CLEANED_DIR}\n")

    # raw string -> {source -> mention count}
    raw_counts = defaultdict(Counter)

    for source, (filename, raw_cols) in SOURCES.items():
        path = CLEANED_DIR / filename
        if not path.exists():
            print(f"  {source:<9} FILE NOT FOUND, skipped")
            continue

        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        column = pick_column(df, raw_cols)
        if column is None:
            print(f"  {source:<9} no author column, skipped")
            continue

        mentions = 0
        for value in df[column]:
            for name in split_authors(value):
                raw_counts[name][source] += 1
                mentions += 1
        print(f"  {source:<9} {mentions:>7,} mentions from column {column!r}")

    if not raw_counts:
        print("\nNo author data read. Run this from the repository root.")
        return

    # ---- Per-name map ----
    map_rows = []
    for raw, per_source in raw_counts.items():
        current = current_normalise(raw)
        current = "" if pd.isna(current) else str(current)
        candidate = candidate_normalise(raw)
        for source, count in per_source.items():
            map_rows.append({
                "raw_name": raw,
                "source": source,
                "n_mentions": count,
                "current_key": current,
                "candidate_key": candidate,
                "changed_by_current": int(current.strip() != str(raw).strip()),
            })

    map_df = pd.DataFrame(map_rows).sort_values(
        ["candidate_key", "source", "raw_name"])
    map_df.to_csv(MAP_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(map_df):,} rows to {MAP_PATH}")

    # ---- Grouped review file ----
    groups = defaultdict(lambda: {
        "variants": set(),
        "sources": set(),
        "mentions": 0,
        "current_keys": set(),
        "given_names": set(),
        "middle_initials": set(),
    })

    for raw, per_source in raw_counts.items():
        key = candidate_normalise(raw)
        if not key:
            continue
        group = groups[key]
        group["variants"].add(raw)
        group["sources"].update(per_source.keys())
        group["mentions"] += sum(per_source.values())
        current = current_normalise(raw)
        if not pd.isna(current):
            group["current_keys"].add(str(current).strip())
        full_names, middles = name_parts_in(raw, key)
        group["given_names"].update(full_names)
        group["middle_initials"].update(middles)

    group_rows = []
    for key, group in groups.items():
        given = sorted(group["given_names"])
        middles = sorted(group["middle_initials"])

        if len(given) >= 2:
            flag = "LIKELY DIFFERENT PEOPLE"
            reason = "two or more distinct given names"
        elif len(middles) >= 2:
            flag = "LIKELY DIFFERENT PEOPLE"
            reason = "same given name, different middle initials"
        elif len(given) == 1:
            flag = "PROBABLY SAFE"
            reason = "one given name, no conflicting middle initials"
        else:
            flag = "NO GIVEN NAME"
            reason = "initials only, cannot be judged from strings"

        group_rows.append({
            "candidate_key": key,
            "n_variants": len(group["variants"]),
            "n_mentions": group["mentions"],
            "n_sources": len(group["sources"]),
            "sources": "; ".join(sorted(group["sources"])),
            "review_flag": flag,
            "review_reason": reason,
            "n_distinct_given_names": len(given),
            "distinct_given_names": "; ".join(given),
            "n_distinct_middle_initials": len(middles),
            "distinct_middle_initials": "; ".join(middles),
            "n_current_keys": len(group["current_keys"]),
            "variants": "; ".join(sorted(group["variants"])),
        })

    groups_df = pd.DataFrame(group_rows).sort_values(
        ["n_variants", "n_mentions"], ascending=False)
    groups_df.to_csv(GROUPS_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(groups_df):,} rows to {GROUPS_PATH}")

    # ---- Summary ----
    total = len(groups_df)
    multi = groups_df[groups_df["n_variants"] >= 2]
    flags = groups_df["review_flag"].value_counts()

    print(f"\n{'=' * 70}\nREVIEW SUMMARY\n{'=' * 70}")
    print(f"  proposed standardised names:        {total:,}")
    print(f"  ...merging 2+ raw spellings:        {len(multi):,}")
    print(f"  ...merging 5+ raw spellings:        "
          f"{len(groups_df[groups_df['n_variants'] >= 5]):,}")
    print("\n  review flags:")
    for flag, count in flags.items():
        print(f"    {flag:<26} {count:>7,}  ({100 * count / total:4.1f}%)")

    risky = groups_df[
        (groups_df["review_flag"] == "LIKELY DIFFERENT PEOPLE")
        & (groups_df["n_variants"] >= 2)
    ]
    print(f"\n  merges that need a human decision:  {len(risky):,}")
    print(f"  author mentions affected by those:  "
          f"{int(risky['n_mentions'].sum()):,}")
    print("\n  Sort author_name_groups.csv by review_flag then n_variants and")
    print("  start at the top. Those are the merges that would be wrong.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
