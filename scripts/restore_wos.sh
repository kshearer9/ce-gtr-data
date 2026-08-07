#!/usr/bin/env bash
# Restore the Web of Science collection from 6 August and relabel the
# publications.
#
#     bash restore_wos.sh
#
# Why this exists. The collection that ran this morning at 10:16 is strictly
# smaller than the one from 6 August: 371 records and 14 whole projects are
# missing, and nothing was gained. The losses are concentrated in 2024 to 2026,
# which is the signature of a collection that stopped early rather than one
# that was deliberately filtered. wos_outcomes_latest.csv was pointed at the
# smaller file, which is why the publication count fell from 6,892 distinct
# DOIs to 6,534 in this morning's run.
#
# The 6 August files are intact in the archive, so this restores them. The
# date fix lives in the cleaner rather than the collector, so re-cleaning the
# restored files gives you the full data and the corrected dates together.
#
# The institution file is taken from the 12:12 run, not 20:32. The 20:32 one
# is the 1-byte file left by the interrupted resume.
#
# Do not re-run collect_wos.py until we know why this morning's run came back
# short. It would overwrite _latest again.

set -euo pipefail

PY=/opt/anaconda3/bin/python
W=data/processed/wos
STAMP=20260806_121255

for f in "$W/wos_outcomes_$STAMP.csv" \
         "$W/wos_outcomes_unique_$STAMP.csv" \
         "$W/wos_outcomes_institutions_$STAMP.csv"; do
    [ -s "$f" ] || { echo "MISSING or empty: $f"; exit 1; }
done

echo "before:"
wc -l < "$W/wos_outcomes_latest.csv" | xargs echo "  wos_outcomes_latest.csv lines:"

cp "$W/wos_outcomes_$STAMP.csv"              "$W/wos_outcomes_latest.csv"
cp "$W/wos_outcomes_unique_$STAMP.csv"       "$W/wos_outcomes_unique_latest.csv"
cp "$W/wos_outcomes_institutions_$STAMP.csv" "$W/wos_outcomes_institutions_latest.csv"

echo "after:"
wc -l < "$W/wos_outcomes_latest.csv" | xargs echo "  wos_outcomes_latest.csv lines:"

echo
echo "--- re-cleaning Web of Science ---"
$PY -m scripts.cleaning.clean_wos_outcomes

echo
echo "--- relabelling publications ---"
$PY scripts/classification/label_publications.py

echo
echo "Done. Expect about 6,892 distinct DOIs again rather than 6,534."
