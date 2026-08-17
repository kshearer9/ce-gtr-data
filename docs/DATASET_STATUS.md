# Dataset status, 17 August 2026

Which files are final, which are stale, and which exist only as an audit
trail. Compiled by walking every file under `ce-gtr-data/data/`, checking
modification times against the scripts that consume them, and rebuilding the
analysis layer to see whether anything moved.

**Headline: the data is final.** One stale file was found and fixed on
17 August; see section 2. Everything is now current, internally consistent,
and reproduces the figures committed to the Methodology chapter.

---

## 1. Final and clean, use these

These are the inputs to `Analysis/scripts/build_final_database.py` and the
only files any analysis should read.

| File | Rows | Modified | What it is |
|---|---:|---|---|
| `cleaned/merged/projects.csv` | 1,640 | 16 Aug | the project spine, the input side |
| `cleaned/merged/outcomes.csv` | 21,154 | 16 Aug | harmonised outcome records, 20,302 distinct |
| `cleaned/merged/project_outcome_map.csv` | 21,154 | 16 Aug | project to outcome, with per-source provenance |
| `cleaned/institutions/institutions.csv` | 7,888 | 15 Aug | organisation registry, ROR-resolved |
| `cleaned/institutions/project_institutions.csv` | 2,462 | 15 Aug | project to organisation, long, with role |
| `cleaned/outcomes/gtr_all_outcomes_clean.csv` | 28,177 | 5 Aug | all twelve GtR outcome types |
| `cleaned/outcomes/openalex_all_outcomes_clean.csv` | 3,264 | 16 Aug | OpenAlex publications |
| `cleaned/outcomes/scopus_all_outcomes_clean.csv` | 4,432 | 16 Aug | Scopus publications |
| `cleaned/outcomes/wos_all_outcomes_clean.csv` | 4,733 | 16 Aug | WoS records, author-expanded |
| `cleaned/outcomes/wos_outcomes_unique_clean.csv` | 4,070 | 16 Aug | WoS deduplicated to paper level |
| `cleaned/outcomes/publications_labelled.csv` | 6,534 | 10 Aug | DOI to discipline, the output side |
| `cleaned/outcomes/publication_field_probabilities.csv` | 6,436 | 10 Aug | outcome-side probability distributions |
| `cleaned/authors/author_identities.csv` | 41,637 | 13 Aug | disambiguated author identities |
| `cleaned/authors/authors_long.csv` | 79,964 | 13 Aug | author to outcome, long |
| `crosswalk/crosswalk_gtr_to_openalex_FINAL.xlsx` | 54 subjects | 6 Aug | THE crosswalk |
| `classification/results/*_merged10*`, `threshold_summary.json`, `accuracy_reject_curve.csv`, `multilabel_evaluation.csv`, `oof_predictions_H.csv` | | 16 Aug | the figures the Methodology chapter quotes |
| `validation/` | | 7 to 16 Aug | blind coding, screening validation, verification key |

**The 16 August rebuild changed columns, not rows.** OpenAlex, Scopus and WoS
all hold exactly the same record counts as the 6 and 7 August versions; the
files grew because the merge script was refactored and gained fields such as
`start_year` and `end_year`. Nothing about the sample changed, which is why
rebuilding the analysis layer left every headline figure untouched.

**Discipline coverage is complete.** `publications_labelled.csv` covers 100%
of the distinct DOIs in the current OpenAlex, Scopus and WoS files and 81.4%
of those in the GtR outcomes file, so it did not go stale when the outcomes
were refreshed.

---

## 2. The one file that was stale, now fixed

**`classification/projects_labelled_final.csv` and
`classification/project_field_probabilities.csv`, both were dated 7 August.**

The classification comparison was re-run on 16 August at 18:28, regenerating
`setups_merged10.csv`, `oof_predictions_H.csv`, `accuracy_reject_curve.csv`
and `threshold_summary.json`. The labels were not re-applied afterwards, so
the labelled file and the threshold that was supposed to have produced it
disagreed. The threshold in force is **0.5029**; the threshold implied by the
7 August labelled file was **0.5821**.

Re-applied on 17 August with:

    python scripts/classification/apply_classifier.py \
        --crosswalk merged10 --proba-only --reuse-sample

`--proba-only` reuses the recorded threshold and weight rather than
re-deriving them, so the evaluation artefacts are untouched. `--reuse-sample`
keeps the projects already in `discipline_verification_KEY.csv`, so coding
already done stays valid. 85 seconds.

### Tier counts, and why three sets of numbers exist

| | Tier 1 (funder) | Tier 2 (model, confident) | Tier 3 (model, below threshold) |
|---|---:|---:|---:|
| `projects_labelled_final.csv`, 7 Aug, superseded | 299 | 724 | 617 |
| `threshold_summary.json`, 16 Aug, historical | 299 | 872 | 469 |
| **current, 17 Aug** | **299** | **844** | **497** |

The 16 August and 17 August figures differ because `apply_classifier.py`
reads its project text from `cleaned/merged/projects.csv`, and that file was
rewritten at 19:06 on 16 August, after the 18:28 classifier run. The 872 / 469
split was therefore a refit against the previous version of the project file.
The model is otherwise deterministic, so this is the only moving part.

**The threshold is unaffected by that.** `_oof_and_threshold` derives it from
the gold set and the training corpus and never reads `merged/projects.csv`,
so 0.5029 remains valid. What is stale is only the `tier_counts` block inside
`threshold_summary.json`, which is a by-product of the refit rather than a
result. **Read that block as historical and quote the database instead.**

The 17 August figures are the first in which the labels and the database
spine were built from the same version of the project file.

**What changed and what did not.** The assigned discipline is the argmax of
the probability distribution, so `discipline_field` is unchanged: Engineering
794, Chemical Engineering 243, Energy 178, Environmental Science 118, and so
on. Only `discipline_tier` moved. Confident (tier 1 or 2) coverage now stands
at 76.8% for university-led projects and 62.1% for company-led ones, which is
worth reporting, since company abstracts are shorter and the classifier is
measurably less sure about them.

---

## 3. Stale, and not used by anything final

| File | Why |
|---|---|
| `cleaned/openalex_projects_clean.csv` (10,334 rows, 31 Jul) | superseded by `processed/openalex/openalex_projects_latest.csv` (11,814 rows, 16 Aug). Nothing downstream reads it |
| `FINAL/` folder | a 30 July to 5 August snapshot. `FINAL/2_collection/gtr_projects_latest.csv` is from the 1,380-project era, and `FINAL/3_gold_standard/gold_james.csv` predates the 6 August rebuild. Its own `INDEX.md` says it is a view, not a store, and can be deleted and rebuilt. **Do not cite anything from it** |
| `processed/wos/wos_outcomes_institutions_20260806_203246.csv` | zero rows, a failed run |
| `cleaned/authors/authors_standardised_excel.csv` | an Excel-safe variant of `authors_standardised.csv`, not a separate dataset |

---

## 4. Superseded, keep as audit trail, never analyse

Already quarantined and correctly named. Left in place because they are the
record of how the sample was arrived at.

- `cleaned/_superseded_1380/` — the pre-filter-redesign 1,380-project state
- `cleaned/_superseded_union/` — the 1,673-project union variant
- `cleaned/gtr_projects_union.csv`, `cleaned/recovered_from_round1.csv`
- `processed/gtr_backup_20260806/` — the full pre-redesign backup
- `crosswalk/crosswalk_gtr_to_openalex_FINAL_pre_6aug.xlsx`
- `cleaned/outcomes/gtr_publications_clean.csv` — 258 projects against the
  stacked file's 330, and 10 project ids absent from the spine. Already
  flagged in `cepaths.py`
- `classification/gold_labelled_projects.csv`, `gold_james.csv`,
  `gold_kirsty.csv`, `cv_folds_FROZEN.json`, and every result file without
  `merged10` in the name — superseded by the merged-crosswalk variant, but
  needed for the sensitivity analysis reported in the chapter
- every `*_latest.csv` and timestamped snapshot under `processed/` — the
  cleaned layer is what analysis reads

---

## 5. Provenance, large, keep, never analyse

| Path | Size | What it is |
|---|---:|---|
| `data/raw/*.jsonl` | 4.5 GB | every API response, the collection audit trail |
| `processed/gtr/gtr_all_with_decision_20260806_202646.csv` | 409 MB | every candidate project with its screening decision, the PRISMA audit trail for the current 1,640 |
| `processed/gtr/gtr_all_with_decision_20260730_112933.csv` | 141 MB | the same for the superseded 1,380 |
| `classification/gtr_corpus_labelled.csv` | 143 MB | the subject-tagged training corpus |
| `classification/*_embeddings*.npy` | 251 MB | mpnet and MiniLM embeddings |

The repository is 9.4 GB, of which `data/` is 6.7 GB and `data/raw/` alone is
4.5 GB. All of it is gitignored.

---

## 5a. Provenance of the merged layer, and three known defects

Recovered from a README left in the now-deleted `Diss/analysis (old)` folder,
recorded here because it exists nowhere else.

> Snapshot from Kirsty's outcome-merge branch (commit `c716aed`), 16 Aug 2026.
> PROVISIONAL. Only exists because of four local patches to her code that are
> not in the repo: the team-code path in `project_outcome_mapping.py`,
> `parse_author` in `collect_scopus_outcomes.py`, and the `keywords_plus` JSON
> path in `collect_wos.py`. Also predates her outstanding fixes.

**This matters for reproducibility.** Those three files still show as modified
and uncommitted in `git status`. Until they are committed, nobody else,
including Kirsty, can regenerate the merged layer the whole database rests on.
Commit them.

The README also flagged three data defects. All three were checked against the
database:

1. **`cited_by` ignores Web of Science.** Confirmed, and it survived the
   17 August merge rewrite. That rewrite states its rule as "first populated
   source (WoS then Scopus)" but reports `Sources Combined: 1` and contributes
   no WoS values. In the current data 4,069 outcomes carry
   `wos_times_cited_all_db`, 1,942 of those also have `cited_by`, and **2,127
   have a WoS count with `cited_by` left null**. If WoS were genuinely first,
   all 4,069 would come through. Raised with Kirsty as a question rather than
   asserted as a bug, since it was diagnosed from the merge summary and the
   row counts rather than from reading her merge code.

   The same rewrite deliberately dropped OpenAlex from `cited_by`, so raw
   coverage fell from 5,291 publications to 4,102. The build's `cited_by_best`
   recovers the WoS values and brings it to **6,219 of 9,612 publications,
   65%**. That is still 417 below the pre-rewrite figure, because OpenAlex-only
   citations are now genuinely absent. `openalex_cited_by` is retained as its
   own column if a sensitivity analysis wants it.
2. **"9,551 outcomes have no title."** True but misleading. Of those, 9,508
   are disseminations, collaborations, further fundings and policy influences,
   which GtR records without a title. **Only 5 publications lack one.**
3. **"9,973 abstracts are GtR descriptions rather than real abstracts."**
   Exactly right, and the 17 August merge confirms the figure: 9,973 of 17,376
   abstracts are GtR descriptions. It sounds fatal for topic modelling and is
   not, because the contamination sits almost entirely in the non-publication
   types. Of 9,612 publications, 6,502 have an abstract and **6,428 of those
   also have a DOI**, so only 74 are description-derived. The usable RQ2
   corpus is about 6,500 publications, and the constraint is missing abstracts
   rather than wrong ones. Year is present for 7,501 publications and runs
   2006 to 2026, so roughly 2,100 drop out of any time series.

4. **Year is merged by source priority, not majority.** Kirsty's note
   describes a majority vote; the merge reports "first populated source (GtR
   then Scopus then WoS then OpenAlex)" and its outlier counters read zero
   across all 3,842 disagreements. **Do not describe it as a majority vote in
   the methodology.** The behaviour itself is fine and should not be changed:
   GtR against WoS has a median difference of zero and only 28 records differ
   by more than a year.

## 6. Two conventions worth knowing before you write the Results chapter

**Grain of the merged outcome file.** `merged/outcomes.csv` is row-for-row
aligned with `merged/project_outcome_map.csv`, so it is at the
project-outcome-*link* grain, not the outcome grain. 612 outcomes are linked
to more than one project and appear more than once. Deduplicate on
`global_outcome_id` before counting papers. The database does this for you:
`outcomes` holds 20,302 distinct records, `project_outcomes` holds the 21,154
links.

**Two publication counts, both correct.** The source-stacked definition, the
one the Methodology chapter quotes, gives **616** projects with a
publication. The merged deduplicated definition gives **615**. The difference
is one project whose only publication record carries no DOI and no title
match, so the merge could not place it. Both are in the database
(`has_publication` and `has_publication_merged`) so the chapter's figure
stays reproducible.

**One outcome is typed inconsistently.** A single record is a dataset in GtR
and a conference paper in Scopus. It is flagged `type_ambiguous` in the
`outcomes` table rather than silently resolved.

**WoS row counts are author-expanded.** `wos_all_outcomes_clean.csv` has
4,733 rows against `wos_outcomes_unique_clean.csv`'s 4,070 papers. The
database counts WoS publications with `COUNT(DISTINCT doi)`, so the counts are
right, but `source_outcome_links` carries 663 more WoS rows than there are WoS
papers. Do not count rows in that table.

---

## 7. Verification

Rebuilt on 17 August after the classifier was re-applied. The database
reproduces every figure committed to the Methodology chapter:

| Figure | Rebuilt | Chapter | |
|---|---:|---:|---|
| Projects | 1,640 | 1,640 | matches |
| With a publication | 616 | 616 | matches |
| Without a publication | 1,024 | 1,024 | matches |
| Total funding | £752.0M | £752M | matches |

Structural checks all pass: no duplicate primary keys in `projects`,
`outcomes` or `institutions`; no duplicate project-outcome pairs; no orphan
rows in either link table; `PRAGMA integrity_check` returns `ok`.

Re-applying the discipline labels moved no output figure, which is the
expected result: discipline is joined onto the project spine and plays no
part in how outcomes are counted. Had any of the four figures moved, it would
have meant the discipline join was fanning out rows.

**The data phase is closed.** Every file feeding the database is current, and
`BUILD_MANIFEST.md` carries a SHA-256 for each one, so any figure in the
Results chapter can be traced to the exact bytes behind it.
