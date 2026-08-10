"""Establish how the GtR search endpoint parses queries.

Section 3.3 of the methodology makes three empirical claims about the endpoint:
that an unquoted multi-word term is evaluated as a disjunction of its
constituent words rather than as a phrase, that truncation with `*` is
supported, and that Boolean OR is supported. None of this is stated in the API
documentation, and the endpoint is described only as a keyword search, so each
claim is an assumption until probed. This script probes them.

Every probe is designed so that the expected count differs depending on which
interpretation is true. Nothing here depends on reading the documentation.

    Truncation
        `circular economy` against `circular econom*`
        A working wildcard matches at least as many projects as the literal,
        since "economy" is among the expansions of "econom*". A count at or
        below the literal means `*` is being matched as an ordinary character.

    Boolean OR
        `zqxwvu OR circular economy` is the decisive probe. "zqxwvu" matches
        nothing. If OR is parsed, the query returns roughly the count for
        "circular economy" alone. If it is not, the nonsense token either drags
        the result towards zero under AND semantics, or the word "OR" becomes a
        search token in its own right. The control confirms "zqxwvu" is empty.

    Default multi-word behaviour
        `circular`, `economy`, `circular economy`, `"industrial symbiosis"`
        If the unquoted pair returns about as many as the rarer single word the
        endpoint ANDs by default; if about as many as the commoner one, it ORs.
        The quoted phrase gives the contrast the methodology reports, since the
        gap between quoted and unquoted counts is what justifies leaving
        discrimination to the screening rule rather than to the search.

    The full vocabulary
        The eighteen terms joined with OR, to confirm that a single combined
        query returns the same set as querying each term separately and taking
        the union. The terms are read from the collector rather than duplicated
        here, so the two cannot drift apart.

Run from the repo root:

    /opt/anaconda3/bin/python scripts/collection/diagnose_query_syntax.py

Prints a table and an interpretation. Nothing is written and no projects are
collected: each probe is a single request for page 1.
"""
from pathlib import Path
import ast
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts" / "collection" / "collect_gtr_projects.py"

BASE_URL = "https://gtr.ukri.org/gtr/api/projects"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
}


def load_terms():
    """Read DEFAULT_TERMS out of the collector without importing it.

    The collector opens a database connection at import time, so importing it
    to read one constant would have side effects. Parsing the source instead
    keeps this script side-effect free while guaranteeing it probes the same
    vocabulary the collection actually used.
    """
    tree = ast.parse(COLLECTOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_TERMS":
                    return ast.literal_eval(node.value)
    raise SystemExit(f"DEFAULT_TERMS not found in {COLLECTOR}")


PROBES = [
    ("truncation", "circular economy", "literal, the baseline"),
    ("truncation", "circular econom*", "should be >= the literal if * works"),
    ("truncation", "remanufacturing", "literal"),
    ("truncation", "remanufactur*", "should be >= the literal if * works"),
    ("boolean", "industrial symbiosis", "single term, unquoted"),
    ("boolean", "circular economy OR industrial symbiosis",
     "should exceed both if OR works"),
    ("boolean", "zqxwvu OR circular economy",
     "DECISIVE: ~= circular economy if OR works, ~0 if not"),
    ("boolean", "zqxwvu", "control, should be 0"),
    ("default", "circular", "single word"),
    ("default", "economy", "single word"),
    ("default", "\"industrial symbiosis\"", "quoted phrase"),
]


def total(session, q):
    try:
        r = session.get(BASE_URL, headers=HEADERS,
                        params={"q": q, "p": 1, "s": 10}, timeout=60)
        r.raise_for_status()
        return r.json().get("totalSize")
    except Exception as exc:                      # noqa: BLE001
        return f"ERROR {exc}"


def main() -> None:
    terms = load_terms()
    counts = {}
    with requests.Session() as session:
        group = None
        for grp, q, note in PROBES:
            if grp != group:
                print(f"\n--- {grp} ---")
                group = grp
            n = total(session, q)
            counts[q] = n
            print(f"  {str(n):>8}   {q:<45} {note}")
            time.sleep(1)

        print(f"\n--- the full vocabulary ({len(terms)} terms) ---")
        combined = total(session, " OR ".join(terms))
        print(f"  {str(combined):>8}   {len(terms)} terms joined with ' OR '")

    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)

    lit, wild = counts.get("circular economy"), counts.get("circular econom*")
    if isinstance(lit, int) and isinstance(wild, int):
        verdict = "SUPPORTED" if wild >= lit else "NOT SUPPORTED"
        print(f"Truncation: {verdict} ({wild} against {lit} for the literal).")
        if wild < lit:
            print("  '*' is being matched as an ordinary character, so every "
                  "truncated\n  term is searching for a string that does not occur.")

    dec, ce = counts.get("zqxwvu OR circular economy"), counts.get("circular economy")
    ctrl = counts.get("zqxwvu")
    if isinstance(dec, int) and isinstance(ce, int):
        ok = bool(ce) and dec >= 0.8 * ce
        print(f"Boolean OR: {'SUPPORTED' if ok else 'NOT SUPPORTED'} "
              f"({dec} against {ce} for the term alone, control {ctrl}).")
        if not ok:
            print("  A nonsense token should not reduce the result if OR is "
                  "parsed. The query\n  is probably treated as a bag of words, "
                  "in which case ' OR ' is itself a token.")

    circ, econ, pair = (counts.get("circular"), counts.get("economy"),
                        counts.get("circular economy"))
    if all(isinstance(x, int) for x in (circ, econ, pair)):
        mode = "disjunction" if pair >= max(circ, econ) else "conjunction"
        print(f"Unquoted multi-word terms are evaluated as a {mode}: "
              f"'circular' {circ}, 'economy' {econ},\n  the pair {pair}.")

    unq, quo = counts.get("industrial symbiosis"), counts.get("\"industrial symbiosis\"")
    if isinstance(unq, int) and isinstance(quo, int) and quo:
        print(f"Quoting narrows sharply: 'industrial symbiosis' returns {unq} "
              f"unquoted and {quo}\n  quoted, a factor of {unq/quo:.0f}. This gap "
              f"is why discrimination is left to the\n  screening rule rather "
              f"than to the search.")

    if isinstance(combined, int):
        print(f"\nThe {len(terms)} terms as one OR query return {combined}. Compare "
              f"this with the unique\ncount from the per-term collection: if they "
              f"agree, querying separately is a\nchoice made to preserve term "
              f"provenance rather than a workaround.")


if __name__ == "__main__":
    sys.exit(main())
