#!/usr/bin/env bash
# Rebuild everything from a fresh canonical GtR collection.
#
# Run from the repo root:
#     caffeinate -i bash run_canonical.sh
#
# Replaces the two-round union with a single collection under the final
# eighteen-term vocabulary, then rebuilds every downstream artefact against it:
# cleaning, the merged project table, the three bibliometric outcome sources,
# the publication labels, the embeddings and both crosswalk variants.
#
# Roughly 5 to 6 hours. Ordered so the cheap data-side work finishes early: if
# the run dies during the long modelling stages, everything up to and including
# the labelled publications is already on disk.
#
# The search phase of the collection is NOT checkpointed. If it is interrupted
# in the first 30 to 45 minutes the collection must start over, and rerunning
# this script does exactly that. After the screening summary appears, the
# enrichment phase resumes from its checkpoint.

set -uo pipefail          # NOT -e: handled per stage, so one failure is visible

PY=/opt/anaconda3/bin/python
STAMP=$(date +%Y%m%d_%H%M)
LOG=logs/canonical_${STAMP}.log
mkdir -p logs
exec > >(tee "$LOG") 2>&1

die () { echo; echo "STOPPING: $*"; exit 1; }

echo "=== $(date) : canonical rebuild ==="

# --- 0. Preserve what is about to be replaced -------------------------------
# The union is superseded as the analytical dataset but remains the evidence
# for the search-round comparison reported in the methodology, so it is kept
# rather than overwritten in place.
echo
echo "--- step 0: preserving the superseded union ---"
mkdir -p data/cleaned/_superseded_union
for f in data/cleaned/gtr_projects_union.csv \
         data/cleaned/gtr_projects_clean.csv \
         data/cleaned/merged/projects.csv \
         data/cleaned/recovered_from_round1.csv; do
    [ -f "$f" ] && cp "$f" "data/cleaned/_superseded_union/$(basename "$f")" \
        && echo "  kept $(basename "$f")"
done

# --- 1. Fresh collection ----------------------------------------------------
echo
echo "--- step 1: fresh GtR collection, 18 terms queried separately (1 to 1.75 hrs) ---"
$PY -m scripts.collection.collect_gtr_projects --fresh \
    || die "collection failed. The search phase is not checkpointed, so rerun this script from the top."

# --- 2. Refuse to continue on an implausible N ------------------------------
echo
echo "--- step 2: sanity check on the new project set ---"
$PY - <<'CHECK' || die "the new project count is outside the expected range."
import sys
import pandas as pd
from pathlib import Path

new = pd.read_csv("data/processed/gtr/gtr_projects_latest.csv", dtype=str)
n = len(new)
print(f"canonical collection: {n} projects retained")
if not (1400 <= n <= 1800):
    sys.exit(f"expected between 1400 and 1800, got {n}")

prev = Path("data/cleaned/_superseded_union/gtr_projects_union.csv")
if prev.exists():
    old = pd.read_csv(prev, dtype=str)
    a, b = set(new.project_id), set(old.project_id)
    print(f"previous union     : {len(b)}")
    print(f"  in both          : {len(a & b)}")
    print(f"  lost vs union    : {len(b - a)}")
    print(f"  new vs union     : {len(a - b)}")
print("check passed")
CHECK

# --- 3. Clean and merge -----------------------------------------------------
echo
echo "--- step 3: clean and merge the project table ---"
$PY -m scripts.cleaning.clean_gtr_projects || die "project cleaning failed."
$PY -m scripts.cleaning.merge_datasets || die "merge failed."

$PY - <<'CHECK' || die "the merged table does not match the collection."
import sys
import pandas as pd
raw = len(pd.read_csv("data/processed/gtr/gtr_projects_latest.csv", dtype=str))
mer = pd.read_csv("data/cleaned/merged/projects.csv", low_memory=False)
print(f"merged projects.csv: {len(mer)} rows against {raw} collected")
print(f"carrying research subjects: {mer.research_subjects.notna().sum()}")
if abs(len(mer) - raw) > 5:
    sys.exit(f"merge lost or duplicated rows: {raw} -> {len(mer)}")
print("check passed")
CHECK

# --- 4. Outputs, cheap because the API caches survive ------------------------
echo
echo "--- step 4: re-collect and clean the three outcome sources ---"
for step in "OpenAlex:scripts.collection.collect_openalex" \
            "Scopus:scripts.collection.collect_scopus_outcomes" \
            "WoS:scripts.collection.collect_wos --fresh"; do
    name="${step%%:*}"; mod="${step#*:}"
    echo; echo "  collecting $name"
    $PY -m $mod || echo "  WARNING: $name collection failed, cache intact, rerunnable"
done
for step in "OpenAlex:scripts.cleaning.clean_openalex_outcomes" \
            "Scopus:scripts.cleaning.clean_scopus_outcomes" \
            "WoS:scripts.cleaning.clean_wos_outcomes"; do
    name="${step%%:*}"; mod="${step#*:}"
    echo; echo "  cleaning $name"
    $PY -m "$mod" || echo "  WARNING: $name cleaning failed"
done

echo
echo "--- step 5: label the publications ---"
$PY scripts/classification/label_publications.py || echo "  WARNING: labelling failed"

# --- 6. Embeddings, projects only -------------------------------------------
echo
echo "--- step 6: embed the projects with mpnet (a few minutes) ---"
$PY scripts/classification/embed_texts_mpnet.py --projects-only \
    || die "embedding failed; the classification stages cannot run."

# --- 7. The two crosswalk variants ------------------------------------------
echo
echo "--- step 7: classification, primary crosswalk (about 80 min) ---"
$PY scripts/classification/run_variant.py --crosswalk james \
    || echo "  WARNING: the primary variant failed"

echo
echo "--- step 8: classification, sensitivity crosswalk (about 100 min) ---"
$PY scripts/classification/run_variant.py --crosswalk kirsty \
    || echo "  WARNING: the sensitivity variant failed"

echo
echo "=== $(date) : canonical rebuild finished ==="
echo "Log: $LOG"
