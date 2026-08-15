#!/usr/bin/env bash
# Collect and clean research outputs for the full 1,673-project union.
#
# Run from the repo root:
#     caffeinate -i bash run_outcomes.sh
#
# The bibliometric sources were last collected against the pre-union project
# set, so roughly 315 projects currently have funding records but no outputs.
# All three collectors read data/cleaned/gtr_projects_clean.csv, which is now
# the union, and all three cache their API responses, so this run fetches only
# what is genuinely missing rather than starting over.
#
# Safe to run alongside a modelling job: this is network-bound, that is
# CPU-bound. Safe to interrupt and rerun; the caches and checkpoints resume.

set -uo pipefail          # NOT -e: one source failing must not lose the others

PY=/opt/anaconda3/bin/python
LOG=outcomes_run.log
exec > >(tee "$LOG") 2>&1

echo "=== $(date) : collecting outputs for the union ==="

coverage () {
    $PY - "$1" <<'COV'
import sys, pandas as pd
from pathlib import Path
label = sys.argv[1]
proj = pd.read_csv("data/cleaned/gtr_projects_clean.csv", usecols=["project_id"],
                   low_memory=False)
total = len(proj)
print(f"\n  --- coverage {label} (of {total} projects) ---")
for name, f in [("OpenAlex", "data/cleaned/outcomes/openalex_all_outcomes_clean.csv"),
                ("Scopus",   "data/cleaned/outcomes/scopus_all_outcomes_clean.csv"),
                ("WoS",      "data/cleaned/outcomes/wos_all_outcomes_clean.csv")]:
    if not Path(f).exists():
        print(f"    {name:9s} not yet produced"); continue
    d = pd.read_csv(f, usecols=["project_id"], low_memory=False)
    n = d.project_id.nunique()
    print(f"    {name:9s} {n:>4} projects ({n/total:.0%})")
COV
}

coverage "BEFORE"

# --- collection -------------------------------------------------------------
# Each is allowed to fail without stopping the others: a Scopus quota block or
# a WoS throttle should not cost you the OpenAlex results.
for step in \
    "OpenAlex:scripts.collection.collect_openalex" \
    "Scopus:scripts.collection.collect_scopus_outcomes" \
    "WoS:scripts.collection.collect_wos"
do
    name="${step%%:*}"; mod="${step#*:}"
    echo
    echo "--- collecting $name ---"
    if $PY -m "$mod"; then
        echo "  $name collection finished"
    else
        echo "  WARNING: $name collection failed or was interrupted."
        echo "  Its cache is intact, so rerunning this script resumes it."
    fi
done

# --- cleaning ---------------------------------------------------------------
for step in \
    "OpenAlex:scripts.cleaning.clean_openalex_outcomes" \
    "Scopus:scripts.cleaning.clean_scopus_outcomes" \
    "WoS:scripts.cleaning.clean_wos_outcomes"
do
    name="${step%%:*}"; mod="${step#*:}"
    echo
    echo "--- cleaning $name ---"
    $PY -m "$mod" || echo "  WARNING: $name cleaning failed"
done

coverage "AFTER"

# --- author enrichment ------------------------------------------------------
# Builds an author-level table alongside the outcome-level cleaned files.
#
# It is a separate table because authors cannot live inside an outcome row: the
# cleaned files hold them as one semicolon-joined string, and a delimited
# string cannot carry a per-author ORCID or forename. That is precisely what
# made the Web of Science ORCIDs unusable.
#
# Both steps read the API caches in cache/, so they make no network calls and
# consume no API quota. Nothing in scripts/cleaning/ is modified. The output
# carries the same outcome_id the cleaners produce, so it joins directly onto
# the cleaned outcome files and onto data/cleaned/merged/project_outcome_map.csv.
#
#   harvest_author_identifiers -> data/cleaned/authors/authors_long.csv
#                                 data/cleaned/authors/author_identities.csv
#   standardise_author_names   -> data/cleaned/authors/authors_standardised.csv
#                                 data/cleaned/authors/authors_standardised_excel.csv
#
# Allowed to fail without stopping the run: a missing cache must not cost you
# the cleaned outcome files produced above.
for step in \
    "harvesting author identifiers:scripts.enrichment.harvest_author_identifiers" \
    "standardising author names:scripts.enrichment.standardise_author_names"
do
    name="${step%%:*}"; mod="${step#*:}"
    echo
    echo "--- $name ---"
    $PY -m "$mod" || echo "  WARNING: $name failed"
done

echo
echo "=== $(date) : outputs run finished ==="
echo "Full log in $LOG"
