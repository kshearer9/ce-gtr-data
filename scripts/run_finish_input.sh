#!/usr/bin/env bash
# Finish the input-side classification on the ten-class scheme.
#
# Run from the repo root:
#     caffeinate -i bash scripts/run_finish_input.sh
#
# Two steps, about 2.5 hours total, logged to logs/finish_input_<stamp>.log.
# set -e means a failure in step 1 stops the chain rather than letting step 2
# run against a half-written results table.
#
#   1. run_variant.py --crosswalk merged10
#      Repeats this afternoon's run with the publication field merge applied,
#      which it previously was not, so set-ups B and C dropped 758 publications
#      instead of folding them. Gold set and folds are rebuilt from the same
#      seed, so every row except B and C should reproduce exactly. If H does
#      not come out at 0.611 / 0.716, something upstream changed: stop and say.
#
#   2. apply_classifier.py --crosswalk merged10 --reuse-sample
#      Threshold from the accuracy-reject curve on the ten-class scheme,
#      tiers for all 1,640 projects, the full probability matrix for soft
#      counting, and the verification KEY refreshed for the SAME 100 projects
#      already hand-coded (--reuse-sample), so that coding stays scoreable.
#      Weight selection runs by inner CV because the merged10 run split
#      between x50 and x100 with no clear mode.
#
# NOT included: the output-side relabelling. That patch is being designed and
# tested separately rather than bolted on unattended.

set -euo pipefail

PY=/opt/anaconda3/bin/python
mkdir -p logs
LOG="logs/finish_input_$(date +%Y%m%d_%H%M).log"
exec > >(tee "$LOG") 2>&1

echo "=== $(date) : finishing the input side (merged10) ==="

echo
echo "--- step 1: variant run with B and C corrected (~105 min) ---"
$PY scripts/classification/run_variant.py --crosswalk merged10

echo
echo "--- step 2: threshold, tiers, probability matrix (~45 min) ---"
$PY scripts/classification/apply_classifier.py --crosswalk merged10 --reuse-sample

echo
echo "=== $(date) : input side complete ==="
echo "Check: setups_summary_merged10.csv (B and C now current),"
echo "       threshold_summary.json, projects_labelled_final.csv,"
echo "       project_field_probabilities.csv,"
echo "       discipline_verification_KEY.csv (same 100 projects, new answers)."
echo "Full log in $LOG"
