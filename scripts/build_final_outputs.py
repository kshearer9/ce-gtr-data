"""Gather the current canonical outputs into one folder for easy review.

The repo accumulates timestamped files, superseded runs and parallel variants,
which is correct for an audit trail but hard to read. This script builds a
FINAL/ folder containing only the outputs that are currently authoritative,
plus an INDEX.md saying what each one is and what it supersedes.

FINAL/ is a VIEW, not a store. It is rebuilt from scratch on every run and is
gitignored. Nothing here is a source of truth: if a file in FINAL/ and the same
file in data/ disagree, data/ wins and FINAL/ is stale. Delete it freely.

Large files (the 143MB tagged corpus, the embedding matrices) are listed in the
index with their real path rather than copied, so the folder stays small enough
to open and browse.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/build_final_outputs.py
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "FINAL"
SIZE_LIMIT_MB = 60

# (source path relative to repo root, destination subfolder, description)
COPY = [
    # --- crosswalk -----------------------------------------------------
    ("data/crosswalk/crosswalk_gtr_to_openalex_FINAL.xlsx", "1_crosswalk",
     "THE crosswalk. 54 GtR subjects to OpenAlex fields. Primary scheme."),
    ("data/crosswalk/crosswalk_kirsty_verified.xlsx", "1_crosswalk",
     "Second coder's independent verdicts. Reported as a sensitivity analysis."),
    ("data/crosswalk/openalex_spotcheck_45.xlsx", "1_crosswalk",
     "Blind spot check of 45 projects, hand coded."),

    # --- collection ----------------------------------------------------
    ("data/processed/gtr/gtr_projects_latest.csv", "2_collection",
     "The CE project set that everything downstream is built on."),

    # --- gold standard -------------------------------------------------
    ("data/classification/gold_james.csv", "3_gold_standard",
     "Gold labels under the primary crosswalk. 276 projects, 11 classes."),
    ("data/classification/gold_kirsty.csv", "3_gold_standard",
     "Gold labels under the second coder's crosswalk. 276 projects, 10 classes."),
    ("data/classification/cv_folds_FROZEN.json", "3_gold_standard",
     "The frozen cross-validation splits. Never regenerate these."),

    # --- results -------------------------------------------------------
    ("data/classification/results/setups_summary_james.csv", "4_results",
     "HEADLINE. Set-ups A to H under the primary crosswalk. H wins at 0.560."),
    ("data/classification/results/setups_summary_kirsty.csv", "4_results",
     "Same eight set-ups under the alternative crosswalk. H wins at 0.548."),
    ("data/classification/results/setups_james.csv", "4_results",
     "Per-split results behind the summary, needed for the paired tests."),
    ("data/classification/results/setups_kirsty.csv", "4_results",
     "Per-split results, alternative crosswalk."),
    ("data/classification/results/bakeoff_james.csv", "4_results",
     "Method bake-off, primary crosswalk."),
    ("data/classification/results/bakeoff_kirsty.csv", "4_results",
     "Method bake-off, alternative crosswalk."),
    ("data/classification/results/variant_comparison.csv", "4_results",
     "Wilcoxon tests within and between variants. From compare_variants.py."),
    ("data/processed/gtr/term_test_results.csv", "4_results",
     "Net-new yield per candidate search term. From test_new_terms.py."),

    # --- logs ----------------------------------------------------------
    ("variant_run.log", "5_logs",
     "Full terminal output of the 12 hour two-variant run."),
]

# Big files that are referenced rather than copied.
REFERENCE = [
    ("data/classification/gtr_corpus_labelled.csv",
     "51,324 subject-tagged UKRI projects, crosswalked. The training corpus."),
    ("data/classification/corpus_embeddings_mpnet.npy",
     "mpnet embeddings for the corpus."),
    ("data/classification/project_embeddings_mpnet.npy",
     "mpnet embeddings for the CE projects."),
    ("data/processed/gtr/gtr_all_with_decision_20260730_112933.csv",
     "Every candidate project with its screening decision. The PRISMA audit trail."),
]


def human(n_bytes: int) -> str:
    mb = n_bytes / 1_000_000
    return f"{mb:.1f} MB" if mb >= 1 else f"{n_bytes / 1000:.0f} KB"


def main() -> None:
    if FINAL.exists():
        shutil.rmtree(FINAL)
    FINAL.mkdir()

    lines = [
        "# FINAL outputs",
        "",
        "Rebuilt by `scripts/build_final_outputs.py`. This folder is a VIEW, not a store.",
        "Everything here is a copy of something in `data/`. If they disagree, `data/` wins.",
        "Delete this folder whenever you like and rebuild it.",
        "",
    ]
    missing, copied = [], 0
    current_section = None

    for rel, section, desc in COPY:
        src = ROOT / rel
        if section != current_section:
            titles = {
                "1_crosswalk": "## 1. Crosswalk",
                "2_collection": "## 2. Collection",
                "3_gold_standard": "## 3. Gold standard",
                "4_results": "## 4. Results",
                "5_logs": "## 5. Logs",
            }
            lines += ["", titles.get(section, f"## {section}"), ""]
            current_section = section
        if not src.exists():
            missing.append(rel)
            lines.append(f"- **{Path(rel).name}** NOT YET PRODUCED. {desc}")
            continue
        dest_dir = FINAL / section
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)
        copied += 1
        lines.append(f"- **{src.name}** ({human(src.stat().st_size)}) {desc}")

    lines += ["", "## 6. Too big to copy, left where they are", ""]
    for rel, desc in REFERENCE:
        src = ROOT / rel
        size = human(src.stat().st_size) if src.exists() else "missing"
        lines.append(f"- `{rel}` ({size}) {desc}")

    lines += [
        "",
        "## What is NOT here, deliberately",
        "",
        "Timestamped intermediates, superseded runs (the MiniLM-only results, the "
        "pre-corpus bake-off), checkpoint files and API caches all stay in `data/`. "
        "They are the audit trail and should not be deleted, but they are not what "
        "you want to look at day to day.",
        "",
    ]

    (FINAL / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Built {FINAL}")
    print(f"  {copied} files copied, {len(missing)} not yet produced")
    for m in missing:
        print(f"    missing: {m}")
    print(f"  Open {FINAL / 'INDEX.md'} first.")

    gitignore = ROOT / ".gitignore"
    if gitignore.exists() and "FINAL/" not in gitignore.read_text():
        with gitignore.open("a", encoding="utf-8") as fh:
            fh.write("\n# Generated review folder, rebuild with scripts/build_final_outputs.py\nFINAL/\n")
        print("  Added FINAL/ to .gitignore")


if __name__ == "__main__":
    sys.exit(main())
