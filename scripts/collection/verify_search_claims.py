"""Verify the two empirical claims made in Section 3.3 about the search.

Section 3.3 asserts that a phrase-restricted search would miss projects that the
screening rule retains. That assertion needs a number, and the number depends on
where you look. Counting only the title and abstract overstates the case,
because the GtR endpoint also indexes the technical abstract, the potential
impact statement and the classification fields, none of which are kept in the
master output file. This script therefore reads the raw API responses, which
retain every field as returned.

It answers two questions:

    1. Of the retained projects, how many contain none of the search terms
       verbatim anywhere in the record, rather than merely in title or
       abstract? This is the defensible version of the claim.

    2. Where does each match actually occur? If a term appears only in a
       classification field rather than in free text, the project would still
       have been reachable by a phrase search, and the claim weakens
       accordingly.

Run from the repo root:

    /opt/anaconda3/bin/python scripts/collection/verify_search_claims.py

It streams the raw JSONL files rather than loading them, so memory use stays
low despite the files totalling several gigabytes. Expect a few minutes.

The separate question of what a quoted query returns is answered by
diagnose_query_syntax.py, which probes the API directly.
"""
from pathlib import Path
import argparse
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROJECTS = ROOT / "data" / "cleaned" / "merged" / "projects.csv"

# The eighteen search terms, with the trailing wildcard stripped: a verbatim
# check on "circular econom" catches economy, economies and economic alike,
# which is what the wildcard was for.
TERMS = [
    "circular econom", "industrial symbiosis", "urban min", "remanufactur",
    "circular bioeconom", "cradle-to-cradle", "closed loop", "circular business",
    "circular product", "circular industry", "circular management",
    "circular supply chain", "circular transition", "circular value chain",
    "regenerative design", "regenerative econom", "waste recovery",
    "waste renewal",
]

# Every field in the raw record that carries searchable text. techAbstractText
# and potentialImpact are the two that matter here: they are indexed by the
# endpoint but dropped from the master output, so a check against the cleaned
# file cannot see them.
TEXT_FIELDS = ["title", "abstractText", "techAbstractText", "potentialImpact"]
CLASS_FIELDS = ["researchSubjects", "researchTopics", "healthCategories",
                "researchActivities", "rcukProgrammes"]


def flatten(value):
    """Collapse a nested classification block to its text."""
    out = []
    if isinstance(value, dict):
        for v in value.values():
            out.append(flatten(v))
    elif isinstance(value, list):
        for v in value:
            out.append(flatten(v))
    elif value is not None:
        out.append(str(value))
    return " ".join(out)


def contains(text, terms):
    low = text.lower()
    return [t for t in terms if t in low]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="gtr_raw_*.jsonl",
                    help="which raw files to stream")
    args = ap.parse_args()

    if not PROJECTS.exists():
        sys.exit(f"Missing {PROJECTS}.")
    proj = pd.read_csv(PROJECTS, low_memory=False)
    retained = set(proj.project_id)

    # Step 1: the title-and-abstract-only check, reproducing the figure that
    # the cleaned file supports on its own.
    text = (proj.title_clean.fillna("") + " " + proj.abstract_text_clean.fillna("")).str.lower()
    in_ta = pd.Series(False, index=proj.index)
    for t in TERMS:
        in_ta |= text.str.contains(re.escape(t), regex=True, na=False)
    missing_ta = set(proj.loc[~in_ta, "project_id"])
    print(f"retained projects: {len(proj)}")
    print(f"no term verbatim in title or abstract: {len(missing_ta)} "
          f"({len(missing_ta)/len(proj):.1%})")

    # Step 2: for those, look at everything the API returned.
    files = sorted(RAW_DIR.glob(args.raw_glob))
    if not files:
        sys.exit(f"No raw files matching {args.raw_glob} in {RAW_DIR}.")
    print(f"\nstreaming {len(files)} raw files for the {len(missing_ta)} projects "
          f"with no title or abstract match")

    found = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("id")
                if pid not in missing_ta or pid in found:
                    continue
                free = " ".join(str(rec.get(f) or "") for f in TEXT_FIELDS)
                classif = " ".join(flatten(rec.get(f)) for f in CLASS_FIELDS)
                found[pid] = dict(free=contains(free, TERMS),
                                  classification=contains(classif, TERMS))
        print(f"  {path.name}: {len(found)}/{len(missing_ta)} located", flush=True)

    missing_records = missing_ta - set(found)
    if missing_records:
        print(f"\n  {len(missing_records)} projects not found in the raw files; "
              f"they are excluded from the figures below")

    located = list(found.values())
    if not located:
        sys.exit("No records located. Check --raw-glob.")

    free_only = sum(1 for r in located if r["free"])
    class_only = sum(1 for r in located if not r["free"] and r["classification"])
    nowhere = sum(1 for r in located if not r["free"] and not r["classification"])

    print(f"\n=== of the {len(located)} located ===")
    print(f"  term appears in another free-text field "
          f"(technical abstract or potential impact): {free_only}")
    print(f"  term appears only in a classification field: {class_only}")
    print(f"  term appears nowhere in the record:          {nowhere}")

    total = len(proj)
    print(f"\n=== the defensible figure for Section 3.3 ===")
    if missing_records:
        # Reporting nowhere/total while some records were never located would
        # understate the figure, because an unlocated project is unknown rather
        # than known to contain a term. Refuse to give a headline number.
        print(f"  INCOMPLETE: {len(missing_records)} of {len(missing_ta)} projects "
              f"were not found in the\n  raw files, so no headline figure is given. "
              f"Check that data/raw holds every\n  gtr_raw_*.jsonl from the "
              f"reported collection run, then rerun.")
        print(f"  Provisional, over the {len(located)} located only: "
              f"{nowhere} with no term anywhere ({nowhere/len(located):.1%} of located).")
    else:
        print(f"  no search term anywhere in the record: {nowhere} of {total} "
              f"({nowhere/total:.1%})")
        print(f"  compare with the title-and-abstract-only figure: "
              f"{len(missing_ta)} ({len(missing_ta)/total:.1%})")
        print("\nQuote the first of these. The second counts projects a phrase "
              "search could\nstill have reached through fields the master file "
              "does not keep.")

    out = ROOT / "data" / "classification" / "results" / "search_claim_verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(measure="retained_projects", value=total),
        dict(measure="no_term_in_title_or_abstract", value=len(missing_ta)),
        dict(measure="of_those_term_in_other_free_text", value=free_only),
        dict(measure="of_those_term_in_classification_only", value=class_only),
        dict(measure="no_term_anywhere_in_record", value=nowhere),
        dict(measure="projects_not_located_in_raw", value=len(missing_records)),
    ]).to_csv(out, index=False)
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
