"""Test whether the GtR search endpoint actually supports OR and wildcards.

PR #46 changed Stage 1 from five separate searches to a single query built by
joining nineteen terms with " OR ", several of them using `*` wildcards. That is
only a valid description of what the code does if the GtR API parses boolean
operators and wildcards. Its documentation does not say that it does, and the
endpoint is a plain keyword search, so this is an assumption rather than a fact
until it is tested.

The test is a set of probes whose expected counts differ depending on which
interpretation is true. Nothing here depends on reading the API docs.

    Wildcards
        `circular economy` vs `circular econom*`
        A working wildcard should match at least as many projects as the
        literal, since "economy" is one of the things "econom*" expands to.
        A count of zero, or a count far below the literal, means `*` is being
        treated as an ordinary character.

    Boolean OR
        `circular economy` vs `circular economy OR industrial symbiosis`
        A working OR should return the union, so more than either term alone.

        `zqxwvu OR circular economy`
        The decisive probe. "zqxwvu" matches nothing. If OR is parsed, this
        returns roughly the "circular economy" count. If OR is not parsed, the
        nonsense token drags the result toward zero (under AND semantics) or
        the literal word "OR" becomes a search term in its own right.

    Default multi-word behaviour
        `circular`, `economy`, `circular economy`
        If the pair returns about as many as the rarer single word, the
        endpoint ANDs by default. If it returns about as many as the commoner
        one, it ORs by default. This matters because the nineteen-term query is
        unquoted, so phrases like "industrial symbiosis" may not be held
        together.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/collection/diagnose_query_syntax.py

Prints a table and an interpretation. No data is written and no projects are
collected; each probe is a single request for page 1.
"""
import sys
import time

import requests

BASE_URL = "https://gtr.ukri.org/gtr/api/projects"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "DurhamMDS-CE-ResearchProject/1.0 (academic use)",
}

# PR #46's full term list, to check the combined query at the end.
PR46_TERMS = [
    "circular econom*", "industrial symbiosis", "urban min*", "remanufactur*",
    "circular bioeconom*", "cradle-to-cradle", "closed loop", "circular business",
    "circular product", "circular industry", "circular management",
    "circular value chain", "circular transition", "circular supply chain",
    "waste recovery", "waste renewal", "regenerative design",
    "regenerative econom*",
]

PROBES = [
    ("wildcard", "circular economy", "literal, the baseline"),
    ("wildcard", "circular econom*", "should be >= the literal if * works"),
    ("wildcard", "remanufacturing", "literal"),
    ("wildcard", "remanufactur*", "should be >= the literal if * works"),
    ("boolean", "industrial symbiosis", "single term"),
    ("boolean", "circular economy OR industrial symbiosis",
     "should exceed both if OR works"),
    ("boolean", "zqxwvu OR circular economy",
     "DECISIVE: ~= circular economy if OR works, ~0 if not"),
    ("boolean", "zqxwvu", "control, should be 0"),
    ("default", "circular", "common single word"),
    ("default", "economy", "common single word"),
    ("default", "\"industrial symbiosis\"", "quoted phrase, for comparison"),
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

        print("\n--- the combined PR #46 query ---")
        combined = " OR ".join(PR46_TERMS)
        n_combined = total(session, combined)
        print(f"  {str(n_combined):>8}   19 terms joined with ' OR '")

    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)

    lit, wild = counts.get("circular economy"), counts.get("circular econom*")
    if isinstance(lit, int) and isinstance(wild, int):
        if wild >= lit:
            print(f"Wildcards: LOOK SUPPORTED ({wild} >= {lit}).")
        else:
            print(f"Wildcards: LOOK UNSUPPORTED ({wild} < {lit}). '*' is "
                  "probably being matched as a literal character, so every "
                  "wildcard term in #46 is searching for something that does "
                  "not exist.")

    dec, ce = counts.get("zqxwvu OR circular economy"), counts.get("circular economy")
    if isinstance(dec, int) and isinstance(ce, int):
        if ce and dec >= 0.8 * ce:
            print(f"Boolean OR: LOOKS SUPPORTED ({dec} vs {ce} for the term "
                  "alone).")
        else:
            print(f"Boolean OR: LOOKS UNSUPPORTED ({dec} vs {ce}). A nonsense "
                  "token should not reduce the result if OR is parsed. The "
                  "query is probably being treated as a bag of words, in which "
                  "case ' OR ' is itself a search token.")

    print("\nWhatever the verdict, record it. If either feature is "
          "unsupported, the methodology cannot describe Stage 1 as a boolean "
          "query, and the five-separate-searches approach is the honest "
          "description of what actually retrieves the projects.")


if __name__ == "__main__":
    sys.exit(main())
