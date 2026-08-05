# Discipline classification

Assigns an OpenAlex research field to every CE project, so that funded
discipline (inputs) can be compared with published discipline (outputs) for RQ3.

The problem this stage solves: only 284 of the 1,380 CE projects carry a funder
research-subject tag, and the missingness is structural rather than random
(Innovate UK tags 0% of its 512 projects; research councils tag 35-70%). The
remaining projects need model-assigned labels trained on a small, imbalanced
gold set.

Scripts are listed in run order. Each says what it produces and where its
numbers appear.

## 1. Build the training corpus

**`collect_gtr_tagged_corpus.py`**
Paginates the whole GtR project index, keeps only subject-tagged projects, and
excludes every CE project so the corpus stays disjoint from the evaluation set.
Same API headers, retry/backoff and SQLite cache as the main collector.
Produces `gtr_tagged_corpus.csv`, 51,324 projects.

**`apply_crosswalk_to_corpus.py`**
Applies the verified GtR-to-OpenAlex crosswalk, plus a 28-entry extension
covering research subjects that appear only outside the CE set.
Produces `gtr_corpus_labelled.csv`.

## 2. Embed

**`embed_texts.py`** projects and publications, all-MiniLM-L6-v2 (384-dim).
**`embed_corpus.py`** the wider corpus, same encoder.
**`embed_texts_mpnet.py`** all three sets again with all-mpnet-base-v2
(768-dim), written to `_mpnet`-suffixed filenames so both encoders survive and
the choice of encoder is tested rather than assumed.

## 3. Compare methods and training corpora

Both scripts read the frozen folds in `cv_folds_FROZEN.json` and never
regenerate them. Both take `--suffix _mpnet` to switch encoder.

**`run_bakeoff.py`**
Five methods on identical folds: TF-IDF + logistic regression, TF-IDF + naive
Bayes (the taught COMP42415 baseline), SBERT + logistic regression, nearest
class centroid, and a soft-vote ensemble. Macro-F1 is the primary metric,
declared before any method was run.

**`run_corpus_setups.py`**
Training-corpus comparison, set-ups A to G. Includes the nested
cross-validation that selects the instance-weighting factor on inner folds
only, so the reported figure is not inflated by selection on the test fold.

Together these produce the first three points of the reported progression:
0.429 (projects only), 0.476 (corpus, weighted), 0.522 (mpnet).

## 4. The full run, per crosswalk variant

**`run_variant.py`**
The overnight script. Rebuilds the gold set and folds from a named crosswalk
(`--crosswalk james` or `--crosswalk kirsty`), remaps the corpus, then runs the
bake-off and set-ups A to H. Set-up H is the ensemble and the current best at
**0.560 macro-F1, 69.5% accuracy**. Roughly 4 hours per variant.

**`compare_variants.py`**
Wilcoxon signed-rank tests on matched splits, within each variant (does H beat
G and A?) and between variants (does either crosswalk classify better?), plus
the project-level disagreement breakdown and its cost on the outcome side.

## Superseded, kept for provenance

`run_bakeoff.py` and `run_corpus_setups.py` are functionally subsumed by
`run_variant.py`, which reimplements both against a rebuildable gold set. They
are retained because the MiniLM and mpnet figures reported in the methodology
were produced by them directly, and deleting them would break the audit trail
back to those numbers.

## Notes

- Folds are frozen. Every comparison in this stage is paired on the same 25
  splits, so differences are attributable to the method rather than the split.
- Set-ups B and C apply a leakage guard: publications belonging to a project in
  the current test fold are removed from training.
- No API keys are read by any script here except the GtR collector, which needs
  none. Keys live in a gitignored `.env` and are never committed.
