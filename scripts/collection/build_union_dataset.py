"""Union the two GtR search rounds, so neither loses projects the other found.

Two search strategies were run against GtR. Round 1 (July) used five terms as
five separate paginated searches and kept 1,380 projects. Round 2 (August) used
nineteen terms joined into a single boolean query and kept 1,640. Round 2 is not
a superset: a set of projects present in round 1 does not appear in round 2 at
all, including projects with "Circular Economy" in the title, which a query
containing `circular econom*` cannot legitimately miss.

The most likely mechanism is deep pagination. Round 2's query returns ~110,000
results, about 1,100 pages of a relevance-ranked response, and ranked search
endpoints are unstable at that depth: the ordering shifts between requests and
low-ranked items fall through the gaps. Round 1's largest single query was an
order of magnitude shallower. This is a hypothesis about the cause; the loss
itself is a fact, and this script measures it.

Reporting the union of two search rounds is standard practice where a
supplementary search is run (PRISMA 2020, item 7: report all sources and the
date each was searched). It costs no additional collection and loses nothing.

A project counts as recovered only if it is absent from round 2 by project_id
AND by grant reference AND by normalised title, so projects that merely changed
identifier are not double-counted.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/collection/build_union_dataset.py

Writes:
    data/cleaned/gtr_projects_union.csv     the union, with a search_round column
    data/cleaned/recovered_from_round1.csv  the recovered rows, for inspection
"""
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "cleaned"
ROUND2 = CLEAN / "gtr_projects_clean.csv"
ROUND1 = CLEAN / "_superseded_1380" / "gtr_projects_clean.csv"
OUT = CLEAN / "gtr_projects_union.csv"
RECOVERED = CLEAN / "recovered_from_round1.csv"


def norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def main() -> None:
    for path in (ROUND1, ROUND2):
        if not path.exists():
            sys.exit(f"Missing {path}")

    r2 = pd.read_csv(ROUND2, low_memory=False)
    r1 = pd.read_csv(ROUND1, low_memory=False)
    print(f"round 1 (5 terms):  {len(r1)} projects")
    print(f"round 2 (19 terms): {len(r2)} projects")

    ids = set(r2.project_id)
    refs = set(r2.grant_reference.dropna())
    titles = set(r2.title.map(norm))

    missing = r1[~r1.project_id.isin(ids)].copy()
    same_ref = missing.grant_reference.isin(refs)
    same_title = missing.title.map(norm).isin(titles)
    recovered = missing[~same_ref & ~same_title].copy()

    print(f"\nabsent from round 2 by project_id      : {len(missing)}")
    print(f"  of which reappear under another id   : {len(missing) - len(recovered)}")
    print(f"  genuinely absent, recovered here     : {len(recovered)}")

    # Provenance, so the union can always be split back apart.
    r2 = r2.assign(search_round=["both" if p in set(r1.project_id) else "round2_only"
                                 for p in r2.project_id])
    recovered = recovered.assign(search_round="round1_only")

    # Round 1 predates the matched_search_term -> matched_search_query rename.
    if "matched_search_term" in recovered.columns and "matched_search_query" in r2.columns:
        recovered = recovered.rename(columns={"matched_search_term": "matched_search_query"})

    # The two rounds use different names for the same three fields. Without this
    # the recovered rows carry null value_gbp/start_date/end_date, so any funding
    # total or time series would silently omit them. Renaming rather than adding
    # columns keeps one canonical name per field.
    ALIASES = {"value_pounds": "value_gbp",
               "fund_start": "start_date",
               "fund_end": "end_date"}
    for old_name, new_name in ALIASES.items():
        if old_name in recovered.columns:
            if new_name in recovered.columns:
                recovered[new_name] = recovered[new_name].fillna(recovered[old_name])
            else:
                recovered[new_name] = recovered[old_name]
            recovered = recovered.drop(columns=[old_name])

    # Round 1 carries no gtr_url; it is derivable from the grant reference.
    if "gtr_url" in r2.columns:
        recovered["gtr_url"] = ("https://gtr.ukri.org/projects?ref="
                                + recovered.grant_reference.astype(str))

    union = pd.concat([r2, recovered], ignore_index=True, sort=False)
    assert union.project_id.is_unique, "duplicate project_id in the union"

    union.to_csv(OUT, index=False, encoding="utf-8")
    # A verification link, so the loss can be checked against GtR directly
    # rather than taken on trust.
    ev = recovered.copy()
    ev["verify_url"] = ("https://gtr.ukri.org/projects?ref="
                        + ev.grant_reference.astype(str))
    ev["mentions_circular_economy"] = ev.apply(
        lambda r: bool(re.search(r"circular econom", " ".join(
            str(r.get(c) or "") for c in
            ["title", "abstract_text", "tech_abstract_text", "potential_impact"]),
            re.I)), axis=1)
    cols = [c for c in ["project_id", "grant_reference", "title", "lead_funder",
                        "mentions_circular_economy", "research_subjects",
                        "verify_url"] if c in ev.columns]
    ev.to_csv(RECOVERED, index=False, columns=cols, encoding="utf-8")

    print("\nfield coverage in the union (non-null):")
    for col in ("value_gbp", "start_date", "end_date", "gtr_url",
                "abstract_text", "research_subjects"):
        if col in union.columns:
            n = union[col].notna().sum()
            print(f"  {col:<20} {n:>5} / {len(union)}")

    print(f"\nunion: {len(union)} projects")
    print(union.search_round.value_counts().to_string())
    print(f"\nWrote {OUT}")
    print(f"Wrote {RECOVERED}")
    print("\nNext: rerun the merge so data/cleaned/merged/projects.csv is rebuilt "
          "from the union before any classification work.")


if __name__ == "__main__":
    main()
