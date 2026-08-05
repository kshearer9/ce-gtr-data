"""
inspect_wos_record.py
=====================
One-off diagnostic, run BEFORE writing collect_wos.py.

The probe (test_wos_api.py) confirmed that FG= grant matching works, but the
times-cited value came back empty, which means the parsing path was wrong. This
script answers two questions from real data rather than guesswork:

  1. WHERE does the times-cited value actually sit in the Expanded API response?
     It fetches one full record and walks the entire nested structure, printing
     every path whose key looks citation-related. Whatever it prints becomes the
     parsing path in collect_wos.py.

  2. HOW MUCH QUOTA is left on this key? Clarivate returns per-second, per-day
     and per-year allowances in the response headers. A full run over 337
     projects will consume a chunk of these, so we check headroom before
     committing.

It also saves the complete raw JSON so the record structure can be inspected at
leisure (and shown to Stamos) without spending another API call.

Run from the project root:
    python -m scripts.collection.inspect_wos_record
    python -m scripts.collection.inspect_wos_record --ref EP/S036091/1

Requires WOS_API_KEY in .env (see README > Setup > API keys).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv is not installed. Run: pip install python-dotenv")


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "wos"
RAW_DIR.mkdir(parents=True, exist_ok=True)

WOS_BASE_URL = "https://wos-api.clarivate.com/api/wos"
DATABASE_ID = "WOS"

# Default grant reference: a project the probe confirmed has matches, and whose
# paper is squarely on-topic (a circular economy systems-thinking article).
DEFAULT_REF = "EP/V029746/1"

# Keys worth flagging when walking the response. Citation counts appear under
# various names across the schema, so we cast wide and let the output tell us.
CITATION_HINTS = ("cit", "tc", "times", "count", "silo")


def walk(node, path=""):
    """Yield (path, value) for every leaf in a nested dict/list structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk(value, f"{path}[{i}]")
    else:
        yield path, node


def main():
    parser = argparse.ArgumentParser(description="Inspect one raw WoS record.")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help=f"Grant reference to fetch (default {DEFAULT_REF}).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("WOS_API_KEY")
    if not api_key:
        sys.exit("WOS_API_KEY not found in .env")

    params = {
        "databaseId": DATABASE_ID,
        "usrQuery": f"FG=({args.ref})",
        "count": 1,
        "firstRecord": 1,
    }
    resp = requests.get(
        WOS_BASE_URL,
        params=params,
        headers={"X-ApiKey": api_key, "Accept": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    # ----------------------------------------------------------------- quota
    print("=" * 74)
    print("QUOTA / RATE LIMIT HEADERS")
    print("=" * 74)
    quota_found = False
    for header, value in resp.headers.items():
        low = header.lower()
        if any(k in low for k in ("amount", "remaining", "limit", "quota", "rate")):
            print(f"  {header}: {value}")
            quota_found = True
    if not quota_found:
        print("  (no quota headers returned; check the Clarivate portal instead)")

    # ------------------------------------------------------------ save raw
    out_path = RAW_DIR / f"sample_record_{args.ref.replace('/', '_')}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRaw response saved to: {out_path.relative_to(ROOT_DIR)}")

    records = (
        payload.get("Data", {})
        .get("Records", {})
        .get("records", {})
        .get("REC", [])
    )
    if not records:
        sys.exit("No records returned for that reference; try another --ref.")
    rec = records[0] if isinstance(records, list) else records

    # -------------------------------------------------- citation field hunt
    print("\n" + "=" * 74)
    print("CANDIDATE CITATION FIELDS (paths containing cit/tc/times/count)")
    print("=" * 74)
    hits = [
        (p, v) for p, v in walk(rec)
        if any(h in p.lower() for h in CITATION_HINTS)
    ]
    if hits:
        for p, v in hits:
            preview = str(v)[:60]
            print(f"  {p}\n      = {preview}")
    else:
        print("  (nothing matched; inspect the saved JSON manually)")

    # ------------------------------------------------- top-level structure
    print("\n" + "=" * 74)
    print("RECORD TOP-LEVEL STRUCTURE (for schema design)")
    print("=" * 74)

    def outline(node, prefix="", depth=0, max_depth=3):
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                kind = type(value).__name__
                size = f" [{len(value)}]" if isinstance(value, (list, dict)) else ""
                print(f"  {'  ' * depth}{key} ({kind}{size})")
                outline(value, prefix, depth + 1, max_depth)
        elif isinstance(node, list) and node:
            outline(node[0], prefix, depth, max_depth)

    outline(rec)

    print("\nDone. Paste the two sections above back for the collector build.")


if __name__ == "__main__":
    main()