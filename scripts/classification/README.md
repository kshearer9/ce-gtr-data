# Discipline classification

Assigns a research field to every CE project and every linked publication, so
that funded discipline (inputs) can be compared with published discipline
(outputs) for RQ3.

**The problem this stage solves.** Only 299 of the 1,640 CE projects carry a
funder research-subject tag, and the missingness is structural rather than
random: Innovate UK tags none of its 612 projects, while research councils tag
between 35 and 70 per cent. Comparing inputs with outputs on the labelled fifth
alone would compare funders, not disciplines. The remaining 1,341 projects
therefore need model-assigned labels trained on a small, imbalanced gold set.

**What the stage concluded.** Single-label discipline assignment to
interdisciplinary research is not reliable at the per-project level, and this
was measured rather than assumed. The analysis is consequently built at the
level of aggregation where the estimates are sound: field distributions are
computed from predicted probabilities rather than from hard labels, which was
validated at 0.73 percentage points mean absolute error per field.

---

## Taxonomy

Ten OpenAlex fields. The crosswalk maps 54 GtR research subjects onto OpenAlex
fields; four classes with fewer than five gold cases are dropped from modelling
and two further classes were folded after the twelve-class results showed they
could not be learned at the sizes available:

| Folded | Into | Gold cases | F1 at twelve classes |
|---|---|---|---|
| Biochemistry, Genetics and Molecular Biology | Agricultural and Biological Sciences | 8 | 0.00 |
| Materials Science | Engineering | 11 | 0.27 |

This was a **post-hoc revision**, decided after seeing results. Both schemes are
reported in the methodology and the twelve-class figures are the pre-declared
ones. Training at ten classes was tested against training at twelve and folding
afterwards, and the two were indistinguishable (p = 0.44 macro-F1, p = 1.00
accuracy), so essentially all of the apparent improvement is definitional.

---

## Scripts, in run order

### 1. Build the training corpus

**`collect_gtr_tagged_corpus.py`** paginates the whole GtR project index, keeps
subject-tagged projects, and excludes CE projects so the corpus stays disjoint
from the evaluation set. Produces `gtr_tagged_corpus.csv`, 51,324 projects, of
which 33,061 fall in the modelled classes.

**`apply_crosswalk_to_corpus.py`** applies the crosswalk plus a 28-entry
extension covering subjects that appear only outside the CE set. Produces
`gtr_corpus_labelled.csv`.

### 2. Embed

**`embed_texts_mpnet.py`** embeds projects, publications and the corpus with
all-mpnet-base-v2 and rewrites the row index beside each array. `--projects-only`
skips publications and corpus; `--no-corpus` re-embeds projects and publications
but leaves the corpus alone, which is what you want after an outcome
re-collection.

A stale index against a fresh array is the dangerous failure here: it does not
raise, it silently pairs each row with another row's vector. Every consumer now
checks the lengths and refuses to run on a mismatch.

**`embed_texts.py`** and **`embed_corpus.py`** are the earlier MiniLM versions,
retained because the encoder comparison reported in the methodology was produced
with them.

### 3. Compare methods and training corpora

**`run_variant.py`** is the main run, roughly 100 minutes. It rebuilds the gold
set and folds from a named crosswalk (`--crosswalk james | kirsty | merged10`),
re-maps the corpus, then runs the method bake-off and set-ups A to H on 25
frozen splits.

| Set-up | Trained on | Macro-F1 | Accuracy |
|---|---|---|---|
| A | CE project abstracts only | 0.521 | 62.2% |
| B | publication abstracts only | 0.283 | 38.8% |
| C | projects + publications | 0.393 | 46.3% |
| D | wider GtR corpus only | 0.505 | 61.7% |
| E | corpus + projects, unweighted | 0.510 | 61.7% |
| F | corpus subsampled to the CE class balance | 0.478 | 59.9% |
| G | corpus + projects, CE upweighted | 0.588 | 68.5% |
| **H** | **as G, plus a soft-vote with a TF-IDF classifier** | **0.611** | **71.6%** |

H beats G by +0.023 macro-F1 (p = 0.022) and +0.031 accuracy (p = 0.0003), and
beats A by +0.090 and +0.094 (both p < 0.001), on matched splits.

The weighting factor in G and H is selected by nested cross-validation on inner
folds, so the reported figure is not inflated by selection on the evaluation
fold. A run-time leakage guard drops the 36 corpus rows that are also CE
projects.

**`compare_variants.py`** runs Wilcoxon signed-rank tests on matched splits,
within a variant and between variants, plus the project-level disagreement
breakdown between the two crosswalks.

### 4. Set the threshold and label everything

**`apply_classifier.py`** takes the frozen gold set and folds, rebuilds
out-of-fold predictions from H, reads a confidence threshold off the
accuracy-reject curve against a **target declared in advance**, refits on corpus
plus the whole gold set, and labels the 1,341 projects with no funder subject.

Target 80 per cent accuracy among retained predictions. Threshold 0.582,
retaining 203 of 287 out of fold (70.7%) at 80.3 per cent [74.3, 85.2]. Below
the threshold, 57.1 per cent.

| Tier | n | Share | Meaning |
|---|---|---|---|
| 1 | 299 | 18.2% | funder-assigned, no model involved |
| 2 | 724 | 44.1% | model, at or above threshold |
| 3 | 617 | 37.6% | model, below threshold |

Outputs `projects_labelled_final.csv` and `project_field_probabilities.csv`,
the latter being 1,640 projects by 14 fields with total mass 1,640. Tier 1
projects enter as one-hot because their labels are observed rather than
predicted, including four funder-only fields the model cannot produce.

Useful flags: `--proba-only` reuses the recorded threshold and refits once,
about three minutes; `--reuse-sample` keeps the projects already hand-coded and
refreshes only the model's answers, which matters after a taxonomy change
because the verification sample is stratified by predicted field.

### 5. The output side

**`label_publications.py`** assigns a field to every linked publication through
four routes, ordered by **measured** accuracy rather than by assumption:

| Route | Share | Accuracy |
|---|---|---|
| `direct_openalex`, the paper is in OpenAlex | 39% | 100% by construction |
| `wos_citation_topic`, article-level, derived from overlap | 17% | 80% |
| `abstract_classifier`, TF-IDF over title and abstract | 42% | 72% |
| `scopus_asjc`, journal-level, last resort | 0.1% | 63% |

That ordering inverts what the shared ASJC lineage would suggest. Scopus subject
areas are assigned at journal level, so a paper inherits every area its journal
carries; WoS Citation Topics are assigned per article by citation clustering.
Each derived table is validated by deriving on a random half of the overlap and
scoring on the other half, so the accuracy quoted is out of sample.

6,794 of 6,892 distinct DOIs labelled. Writes
`publication_field_probabilities.csv` on the same principle as the project side:
observed fields one-hot, derived labels contributing the measured distribution
of OpenAlex fields among papers carrying that label, the classifier
contributing its posterior.

The publication confidence tiers should **not** be reported. The rule assigns
tier 2 above a route accuracy of 0.70 and the classifier landed at 0.724, which
moved roughly 2,900 papers across the boundary on a fraction of a point. The
probability matrix is the defensible object.

### 6. Evaluate

**`evaluate_multilabel.py`** scores the classifier against the multi-label truth
already present in the data. GtR tags projects with weighted subjects and the
pipeline keeps only the highest-weighted one; 161 of the 287 gold projects (56%)
span more than one field.

| Definition | Value |
|---|---|
| Strict, top-1 equals the funder's top field | 73.5% |
| Primary or the crosswalk's declared secondary | 76.3% |
| Any field the funder tagged | 87.1% |
| Partial credit, funder's weight on the chosen field | 0.653 of a ceiling of 0.787, 83.1% of achievable |

The model names the funder's top field 74 per cent of the time, a lower-ranked
field they did tag 13 per cent, and something they did not tag 13 per cent.

**`score_verification.py`** scores hand coding of a blind sample against the
model. 100 projects, coded without sight of any model output, then 27 recoded
against the crosswalk conventions after the first pass exposed codebook
ambiguity. Both passes are reported.

Tier 2 agreement 69.8 per cent [56.5, 80.5], top-2 81.1 per cent, kappa 0.562.
The pre-declared 80 per cent target is **neither confirmed nor refuted**: the
point estimate falls short and the interval contains it.

**`score_intercoder.py`** computes three-way agreement once a second coder
returns `discipline_coding_SECOND_CODER.xlsx`: coder against coder, which is the
ceiling, and the model against each. Raw agreement and Krippendorff's alpha.

**`gold_learning_curve.py`** fits set-up H at gold sizes 50 to 229 to test
whether more hand-labelling would help. Fitted gain +0.029 macro-F1 per doubling
against fold-to-fold standard deviation of 0.058. Reaching a 0.05 gain would
need roughly four times the labelled data, so the decision not to label further
rests on measurement rather than assertion.

**`test_soft_counts.py`** compares three ways of estimating the field
distribution, on out-of-fold predictions so nothing is scored on its own
training data:

| Estimator | Total variation | Mean absolute error |
|---|---|---|
| Hard assignment | 0.0906 | 1.51pp |
| Soft counts | 0.0441 | 0.73pp |

Hard assignment overstated Engineering by 6.3 points; soft counting reduces that
to 0.5. On the full dataset it moves Engineering from 49.8 to 40.4 per cent and
Environmental Science from 6.9 to 9.3. **RQ3 is built from the probability
matrices, not from the hard labels.**

The script also contains an `adjusted` estimator that should be ignored: it
solves for the class prior using a confusion matrix estimated on the same
projects whose distribution it recovers, which is circular and returns zero
error by construction regardless of model quality.

---

## Superseded, kept for provenance

`run_bakeoff.py` and `run_corpus_setups.py` are functionally subsumed by
`run_variant.py`. They are retained because the MiniLM and mpnet figures
reported in the methodology were produced by them directly, and deleting them
would break the audit trail back to those numbers.

---

## Notes

- **Folds are frozen.** Every comparison is paired on the same 25 splits, so
  differences are attributable to the method rather than to the split.
- **The leakage guard fires visibly.** Every run prints the 36 corpus rows
  dropped for also being CE projects. A run that does not print it is wrong.
- **Set-ups B and C** apply a second guard: publications belonging to a project
  in the current test fold are removed from training.
- **No API keys** are read by any script here. Keys live in a gitignored `.env`.
- **Reproducibility.** Every figure quoted above is written to
  `data/classification/results/` and committed, so each traces to a file rather
  than to a claim.
