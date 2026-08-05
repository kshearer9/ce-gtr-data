"""Test candidate Stage 1 search terms before committing to them.

Adding a search term widens the net at Stage 1 but does not weaken Stage 2: the
three-tier inclusion rule in collect_gtr_projects.py still decides what is kept.
So the only question a new term has to answer is whether it recovers projects
the current term set misses, and this script answers it directly rather than by
argument.

For each candidate it reports:

    hits         projects the GtR API returns for the term
    pass         of those, how many satisfy the Stage 2 tier rule
    net new      of those, how many are NOT already in gtr_projects_latest.csv
    precision    pass / hits, i.e. how much of the term's yield is usable

A term earns its place on net new. A term with high hits and near-zero net new
is redundant with the existing set and adds only collection time. A term with
low precision is not automatically disqualified, since Stage 2 filters it, but
low precision plus low net new is a clear reject and gives a documented reason
for excluding it (Gusenbauer and Haddaway, 2020, on reporting search strategy
decisions rather than only the final query).

Run from the repo root:
    /opt/anaconda3/bin/python scripts/collection/test_new_terms.py
    /opt/anaconda3/bin/python scripts/collection/test_new_terms.py --terms "waste reuse,renewal"
    /opt/anaconda3/bin/python scripts/collection/test_new_terms.py --max-pages 40

Any page cap that actually bites is reported in the output, so a truncated count
is never mistaken for a complete one.
"""
from pathlib import Path
import argparse
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_gtr_projects import (          # noqa: E402
    BASE_URL, HEADERS, classify_ce, clean_text,
)

ROOT = Path(__file__).resolve().parents[2]
EXISTING = ROOT / "data" / "processed" / "gtr" / "gtr_projects_latest.csv"
OUT = ROOT / "data" / "processed" / "gtr" / "term_test_results.csv"

CANDIDATES = ["waste reuse", "renewal"]
PAGE_SIZE = 100


def fetch_all(term, session, max_pages, delay):
    """Return (projects, truncated) for one search term."""
    out, page, truncated = [], 1, False
    while True:
        r = session.get(BASE_URL, headers=HEADERS,
                        params={"q": term, "p": page, "s": PAGE_SIZE}, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("project", []))
        total_pages = data.get("totalPages", 1)
        if page == 1:
            print(f"  '{term}': {data.get('totalSize')} hits over {total_pages} pages")
        if page >= total_pages:
            break
        if max_pages and page >= max_pages:
            truncated = True
            print(f"  STOPPED at the {max_pages}-page cap; {total_pages - page} "
                  f"pages of '{term}' were not read")
            break
        page += 1
        time.sleep(delay)
    return out, truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=None, help="comma-separated candidates")
    ap.add_argument("--max-pages", type=int, default=0, help="0 means no cap")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    terms = ([t.strip() for t in args.terms.split(",") if t.strip()]
             if args.terms else CANDIDATES)

    if not EXISTING.exists():
        sys.exit(f"Missing {EXISTING}; run collect_gtr_projects.py first.")
    have = set(pd.read_csv(EXISTING, usecols=["project_id"]).project_id)
    print(f"Current CE set: {len(have)} projects\n")

    rows = []
    with requests.Session() as session:
        for term in terms:
            projects, truncated = fetch_all(term, session, args.max_pages, args.delay)
            passed, new_ids, examples = 0, set(), []
            for p in projects:
                include, _ = classify_ce(
                    clean_text(p.get("title")),
                    clean_text(p.get("abstractText")),
                    clean_text(p.get("techAbstractText")),
                    clean_text(p.get("potentialImpact")),
                )
                if not include:
                    continue
                passed += 1
                pid = p.get("id")
                if pid not in have:
                    new_ids.add(pid)
                    if len(examples) < 5:
                        examples.append(clean_text(p.get("title"))[:90])
            hits = len(projects)
            rows.append(dict(term=term, hits=hits, passed_screening=passed,
                             net_new=len(new_ids),
                             precision=round(passed / hits, 3) if hits else 0.0,
                             truncated=truncated))
            print(f"  -> {passed} pass screening, {len(new_ids)} net new")
            for e in examples:
                print(f"       new: {e}")
            print()

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False))
    if frame.truncated.any():
        print("\nWARNING: at least one term was truncated by the page cap. "
              "Rerun without --max-pages before making a decision.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
