"""
merge_enrichment.py
===================
Enrich the API-collected CE dataset with personnel and other fields from the
GtR website bulk CSV exports.

Why this step exists
--------------------
The GtR API search response exposes personnel data (principal investigator
names and, critically, ORCID identifiers) only sparsely. The GtR website bulk
CSV export carries these fields more completely. To support the downstream
disciplinary classification - which links researchers to their publication
records via ORCID - the API dataset is enriched with fields from the website
export.

Method (deterministic, reproducible)
------------------------------------
1. One CSV was downloaded from the GtR website for each of the search terms
   used in the API collection (mirroring the search strategy).
2. The CSVs are concatenated and deduplicated on the GtR project identifier.
3. The combined export is joined onto the API dataset by EXACT MATCH on project
   identifier (a stable unique key - no fuzzy/name matching is used).
4. All website-export columns are carried in with a 'csv_' prefix so they sit
   alongside, and never overwrite, the validated API-collected fields.
5. A boolean column 'csv_matched' flags whether each project was found in the
   website export, so the (small) unmatched residue is transparent.

Inputs  (paths relative to repo root; adjust CSV_DIR/API_PATH if needed):
  - API dataset:        data/processed/gtr_ce_projects_latest.csv
  - Website CSVs:       GtR CSVs/*.csv
Output:
  - data/processed/gtr_ce_projects_enriched.csv

Note on CSV coverage: the final search vocabulary uses five terms (closed-loop
was dropped). The closed-loop website CSV is therefore OPTIONAL here: if the
file is present it is still used (its projects simply may not all match the
current dataset, which is harmless), and if it is absent the script proceeds
without it. The reported match rate is the check that matters - if it is high
(~99%), the available CSVs cover the dataset well and no re-download is needed.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration - paths relative to the repo root (run from there)
# ---------------------------------------------------------------------------
API_PATH = Path("data/processed/gtr_ce_projects_latest.csv")
CSV_DIR = Path("GtR CSVs")
OUT_PATH = Path("data/processed/gtr_ce_projects_enriched.csv")

# The website exports, one per search term. Filenames as downloaded.
# Closed-loop is listed but treated as OPTIONAL (it is no longer a search term);
# any file that is missing is skipped with a warning rather than aborting.
CSV_FILES = [
    "Circular Economy Projects.csv",
    "Industrial Symbiosis Projects.csv",
    "Urban mining projects.csv",
    "Remanufacturing projects.csv",
    "Circular bioeconomy.csv",
]
# Optional legacy CSVs - used if present, silently skipped if not.
OPTIONAL_CSV_FILES = [
    "Closed-loop Projects.csv",
]

# The website export's unique project key, and the API dataset's matching key.
CSV_KEY = "ProjectId"
API_KEY = "project_id"


def main():
    # ---- Load the API dataset ----
    if not API_PATH.exists():
        raise SystemExit(f"API dataset not found at: {API_PATH}\n"
                         f"Run this script from the repo root.")
    api = pd.read_csv(API_PATH)
    print(f"API dataset:            {len(api)} projects")

    # ---- Load and combine the website CSVs ----
    frames = []
    for fn in CSV_FILES:
        path = CSV_DIR / fn
        if not path.exists():
            raise SystemExit(f"Missing required website CSV: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    # Optional CSVs: use if present, warn and skip if not.
    for fn in OPTIONAL_CSV_FILES:
        path = CSV_DIR / fn
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
            print(f"  (optional CSV included: {fn})")
        else:
            print(f"  (optional CSV not found, skipping: {fn})")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Website CSVs combined:  {len(combined)} rows (before dedup)")

    # Deduplicate on the project identifier (same project can appear under
    # more than one search term).
    combined = combined.drop_duplicates(subset=CSV_KEY).reset_index(drop=True)
    print(f"After dedup on {CSV_KEY}: {len(combined)} unique projects")

    # ---- Prefix every website column with 'csv_' so nothing collides with
    #      or overwrites the API-collected fields. The key is handled
    #      separately for the join. ----
    combined = combined.rename(columns={c: f"csv_{c}" for c in combined.columns})
    csv_key_prefixed = f"csv_{CSV_KEY}"

    # ---- Exact-match join on project identifier ----
    api["_join_key"] = api[API_KEY].astype(str)
    combined["_join_key"] = combined[csv_key_prefixed].astype(str)

    enriched = api.merge(
        combined.drop(columns=[csv_key_prefixed]),
        on="_join_key", how="left", validate="m:1",
    )

    # ---- Flag which projects were matched in the website export ----
    # Use a column guaranteed to be present from the CSV side to detect a match.
    probe = "csv_ProjectReference" if "csv_ProjectReference" in enriched.columns else None
    if probe is not None:
        enriched["csv_matched"] = enriched[probe].notna()
    else:
        # Fallback: any csv_ column being non-null indicates a match.
        csv_cols = [c for c in enriched.columns if c.startswith("csv_")]
        enriched["csv_matched"] = enriched[csv_cols].notna().any(axis=1)

    enriched = enriched.drop(columns=["_join_key"])

    # ---- Report coverage ----
    matched = int(enriched["csv_matched"].sum())
    total = len(enriched)
    print(f"\nMatched to website export: {matched}/{total} "
          f"({100*matched/total:.1f}%)")
    print(f"Unmatched (API fields only): {total - matched}")
    if total and matched / total < 0.95:
        print("  WARNING: match rate below 95%. The website CSVs may not fully\n"
              "  cover the current project set - consider re-downloading fresh\n"
              "  CSVs from the GtR website for the five current search terms.")

    def cov(col):
        if col not in enriched.columns:
            return "n/a"
        n = (enriched[col].notna() & (enriched[col].astype(str).str.strip() != "")).sum()
        return f"{n}/{total} ({100*n/total:.0f}%)"

    print(f"\nPersonnel coverage after enrichment:")
    print(f"  csv_PISurname:  {cov('csv_PISurname')}")
    print(f"  csv_PI ORCID iD: {cov('csv_PI ORCID iD')}")

    # ---- Write output ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nEnriched dataset written to: {OUT_PATH}")
    print(f"Columns: {len(enriched.columns)} (API fields + csv_ fields + csv_matched)")
    print("Done.")


if __name__ == "__main__":
    main()