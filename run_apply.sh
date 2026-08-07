#!/usr/bin/env bash
# Refresh the publication labels, then set the confidence thresholds and apply
# the classifier. Run from the repo root:
#
#     caffeinate -i bash run_apply.sh
#
# Roughly 30 to 40 minutes end to end. Everything is logged.
#
# The three steps run in sequence rather than in parallel on purpose. The
# embedding step is only about nine minutes now that --no-corpus skips the
# unchanged 33,000-project corpus, and this machine is fanless, so two heavy
# jobs at once would thermally throttle both and finish later than running
# them one after the other.

set -euo pipefail

PY=/opt/anaconda3/bin/python
mkdir -p logs
LOG="logs/apply_$(date +%Y%m%d_%H%M).log"
exec > >(tee "$LOG") 2>&1

echo "=== $(date) : threshold and application run ==="

# --- 1. Re-embed projects and publications -----------------------------------
# publication_embeddings_mpnet.npy still dates from before the Web of Science
# re-collection, because every run since has used --projects-only. The corpus
# has not changed, so --no-corpus leaves it alone.
echo
echo "--- step 1: re-embed projects and publications (~9 min) ---"
$PY scripts/classification/embed_texts_mpnet.py --no-corpus

# --- 2. Regenerate the publication labels ------------------------------------
echo
echo "--- step 2: relabel the publications (~3 min) ---"
$PY scripts/classification/label_publications.py

# --- 3. Thresholds and final project labels ----------------------------------
# --weight 100 is the modal upweighting factor selected by nested cross-
# validation across the 25 evaluation folds of the canonical run: 100 in 16
# folds, 25 in 4, 50 in 3, 10 in 1 and 1 in 1. Passing it saves re-running the
# inner selection, which would take about 25 minutes to land on the same
# answer. Drop the flag if you would rather have it selected here.
echo
echo "--- step 3: thresholds and final labels (~20 min) ---"
$PY scripts/classification/apply_classifier.py --weight 100

echo
echo "=== $(date) : finished ==="
echo "Full log in $LOG"
echo
echo "Next: code data/validation/discipline_verification_sample.xlsx by hand,"
echo "then score it against discipline_verification_KEY.csv."
