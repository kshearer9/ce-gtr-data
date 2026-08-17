"""
compare_wos_collection.py
=========================
Compare a fresh Web of Science collection against an archived one, and refuse
to bless it if it came back smaller.

Why this exists
---------------
On 7 August a re-run of `collect_wos.py` returned 371 fewer records and
dropped 14 whole projects, with the losses concentrated in 2024 to 2026, which
is the signature of a collection that stopped early rather than one that was
deliberately filtered. Nothing failed and nothing warned. It was found only
because someone counted, and `restore_wos.sh` exists to undo it.

A collection that silently shrinks is the worst kind of failure here, because
every count downstream moves and the cause is invisible by then. So: archive
before collecting, run this after, and only proceed if it passes.

Usage, from the repository root:

    python -m scripts.collection.compare_wos_collection \\
        --archive data/processed/wos/_archive_20260817

Exit code 0 means the new collection is at least as complete as the archive.
Exit code 1 means it is not, and you should restore rather than continue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WOS_DIR = Path("data/processed/wos")

# Files to compare, and the column that identifies a row in each.
TARGETS = [
    ("wos_outcomes_latest.csv", "outcome record"),
    ("wos_outcomes_unique_latest.csv", "distinct paper"),
    ("wos_outcomes_institutions_latest.csv", "institution row"),
]

# A collection is allowed to be marginally smaller without failing: WoS does
# occasionally withdraw a record. More than this and it is a collection fault,
# not editorial churn.
TOLERANCE = 0.01


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # a truncated or 1-byte file from a killed run
        print(f"  [ERROR] {path.name} could not be read: {exc}")
        return None


def compare_one(name: str, label: str, archive_dir: Path) -> bool:
    old = load(archive_dir / name)
    new = load(WOS_DIR / name)
    print(f"\n{label}s  ({name})")
    print("-" * 70)
    if old is None:
        print("  no archived copy, nothing to compare against")
        return True
    if new is None:
        print("  [FAIL] the new file is missing or unreadable")
        return False

    delta = len(new) - len(old)
    floor = len(old) * (1 - TOLERANCE)
    print(f"  archived {len(old):>7,}    new {len(new):>7,}    "
          f"change {delta:+,}")

    ok = len(new) >= floor
    if not ok:
        print(f"  [FAIL] below the {TOLERANCE:.0%} tolerance floor "
              f"of {floor:,.0f}")
    elif delta < 0:
        print(f"  [warn] smaller, but within tolerance")
    return ok


def compare_coverage(archive_dir: Path) -> bool:
    """Projects and DOIs are what actually matter downstream, not row counts."""
    old = load(archive_dir / "wos_outcomes_latest.csv")
    new = load(WOS_DIR / "wos_outcomes_latest.csv")
    if old is None or new is None:
        return True

    ok = True
    print("\nCoverage")
    print("-" * 70)
    for column, label in [("project_id", "projects with at least one record"),
                          ("doi", "distinct DOIs")]:
        if column not in old.columns or column not in new.columns:
            continue
        a = set(old[column].dropna().astype(str).str.lower())
        b = set(new[column].dropna().astype(str).str.lower())
        lost, gained = a - b, b - a
        print(f"  {label}")
        print(f"    archived {len(a):>7,}    new {len(b):>7,}    "
              f"lost {len(lost):,}  gained {len(gained):,}")
        if lost:
            ok = False
            sample = sorted(lost)[:5]
            print(f"    [FAIL] present before and absent now, e.g. {sample}")
    return ok


def citation_column(archive_dir: Path) -> None:
    """Report which citation column each collection carries.

    The whole point of this re-collection is the rename from
    `times_cited_all_db` to `cited_by`, so confirm it actually happened.
    """
    print("\nCitation column")
    print("-" * 70)
    for where, path in [("archived", archive_dir / "wos_outcomes_latest.csv"),
                        ("new", WOS_DIR / "wos_outcomes_latest.csv")]:
        frame = load(path)
        if frame is None:
            continue
        present = [c for c in ("times_cited_all_db", "cited_by",
                               "times_cited_core") if c in frame.columns]
        detail = []
        for column in present:
            values = pd.to_numeric(frame[column], errors="coerce")
            detail.append(f"{column} ({values.notna().sum():,} populated, "
                          f"median {values.median():.0f})")
        print(f"  {where:<9} {', '.join(detail) or 'none found'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True,
                        help="directory holding the archived *_latest.csv files")
    args = parser.parse_args()
    archive_dir = Path(args.archive)
    if not archive_dir.is_dir():
        print(f"archive directory not found: {archive_dir}")
        return 1

    print("=" * 70)
    print("WoS collection comparison")
    print(f"  archive: {archive_dir}")
    print(f"  new:     {WOS_DIR}")
    print("=" * 70)

    results = [compare_one(name, label, archive_dir) for name, label in TARGETS]
    results.append(compare_coverage(archive_dir))
    citation_column(archive_dir)

    print("\n" + "=" * 70)
    if all(results):
        print("PASS. The new collection is at least as complete as the archive.")
        print("Safe to continue with clean_wos_outcomes.py.")
        return 0
    print("FAIL. The new collection lost data relative to the archive.")
    print("Do NOT continue. Restore with:")
    print(f"    cp {archive_dir}/*.csv {WOS_DIR}/")
    print("then re-clean from the restored files.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
