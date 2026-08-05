#!/usr/bin/env bash
# Rebuild the discipline classification pipeline on the 1,673-project union.
#
# Run from the repo root:
#     bash run_rebuild.sh
#
# Roughly 4 to 5 hours. Everything is logged to rebuild_run.log. `set -e` means
# any failing step stops the chain rather than letting a later step run on bad
# input, so if you wake up to a short log, read the end of it: the run stopped
# on purpose.

set -euo pipefail

PY=/opt/anaconda3/bin/python
LOG=rebuild_run.log
exec > >(tee "$LOG") 2>&1

echo "=== $(date) : starting rebuild ==="

# --- 1. Swap the union in, keeping the file it replaces -----------------------
echo
echo "--- step 1: install the union as the cleaned GtR projects file ---"
cp data/cleaned/gtr_projects_clean.csv \
   data/cleaned/_superseded_1380/gtr_projects_clean_1640.csv
cp data/cleaned/gtr_projects_union.csv data/cleaned/gtr_projects_clean.csv
echo "backed up the 1,640 file, installed the 1,673 union"

# --- 2. Rebuild the merged project table -------------------------------------
echo
echo "--- step 2: merge GtR with OpenAlex project metadata ---"
$PY -m scripts.cleaning.merge_datasets

# --- 3. Refuse to continue on the wrong row count ----------------------------
# A silent drop here would poison everything downstream, and the run would look
# like it had succeeded.
echo
echo "--- step 3: check the merged table ---"
$PY - <<'CHECK'
import sys
import pandas as pd

df = pd.read_csv("data/cleaned/merged/projects.csv")
n = len(df)
subjects = df.research_subjects.notna().sum()
print(f"projects.csv rows: {n} (expected 1673)")
print(f"carrying research subjects: {subjects} (expected about 318)")
if n != 1673:
    sys.exit(f"STOPPING: expected 1673 rows, got {n}. The merge dropped or "
             f"duplicated projects, so nothing downstream would be valid.")
print("check passed")
CHECK

# --- 4. Re-embed the projects (mpnet), corpus untouched ----------------------
echo
echo "--- step 4: embed projects with mpnet (~30 min) ---"
$PY scripts/classification/embed_texts_mpnet.py --projects-only

# --- 5. The full variant run --------------------------------------------------
# Watch for the leakage-guard line: the corpus was built excluding the old CE
# set, so projects added since would otherwise train and test at once.
echo
echo "--- step 5: bake-off and set-ups A to H (~4 hours) ---"
$PY scripts/classification/run_variant.py --crosswalk james

echo
echo "=== $(date) : rebuild finished ==="
echo "Results in data/classification/results/"
