"""Would more hand-labelled projects actually improve the classifier?

Before committing a day to coding several hundred more projects, measure
whether performance is still climbing at the gold set size you already have.
This fits set-up H repeatedly with the gold training set subsampled to
increasing sizes, holding the wider GtR corpus, the folds, the crosswalk and
the upweighting factor fixed, so the only thing varying is how many
funder-labelled CE projects the model sees.

A straight line is then fitted to macro-F1 against log2 of the gold set size
and used to project what a larger gold set would buy. A power law would be the
textbook choice, but with five points its three parameters are badly
under-determined, and on test curves it ran to a ceiling of 1.0 and projected
nonsense. The log-linear fit assumes the current rate of improvement simply
continues, which learning curves never do, so its projection is an upper bound.
That is the useful direction to be wrong in: if even the optimistic estimate
says more labelling will not help, it will not help.

This is the data-acquisition use of learning curves, the same framing already
cited in run_learning_curve.py; verify that reference before it goes in the text.

Read the output like this. If the curve is still climbing steeply at your
current size, labelling more projects is the right investment. If it has
flattened, it is not, and the honest options are coarser classes or reporting
the figure you have. Compare any projected gain against the fold-to-fold
standard deviation printed beside it, because a gain smaller than the noise is
not a gain.

Run from the repo root, after run_variant.py --crosswalk james:

    caffeinate -i /opt/anaconda3/bin/python \\
        scripts/classification/gold_learning_curve.py

About 35 minutes at the default five sizes and five folds. Use --folds 3 to cut
it to roughly 20 minutes at the cost of a noisier curve.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import vstack as spvstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
RESULTS = DIR / "results"
PROJECTS = ROOT / "data" / "cleaned" / "merged" / "projects.csv"
CORPUS = DIR / "gtr_tagged_corpus.csv"

SEED = 20260803
C_TFIDF, C_EMB, MAX_ITER = 1.0, 3.0, 3000
VARIANT = "james"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_variant import load_mapping, primary_field  # noqa: E402


def stratified_subset(y, idx, n, rng):
    """n indices from idx, keeping the class balance and one of each class."""
    idx = np.asarray(idx)
    classes, counts = np.unique(y[idx], return_counts=True)
    if n >= len(idx):
        return idx
    take = np.maximum(1, np.floor(counts / counts.sum() * n).astype(int))
    # Trim or pad the largest classes until the total is exactly n.
    order = np.argsort(-counts)
    i = 0
    while take.sum() != n and i < 10_000:
        j = order[i % len(order)]
        if take.sum() < n and take[j] < counts[j]:
            take[j] += 1
        elif take.sum() > n and take[j] > 1:
            take[j] -= 1
        i += 1
    out = []
    for cls, k in zip(classes, take):
        pool = idx[y[idx] == cls]
        out.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.asarray(out)


def log_linear_projection(sizes, scores):
    """Least squares fit of score against log2(size).

    Deliberately the primary estimate rather than a power law. With five points
    a three-parameter power law is badly under-determined; on test curves it
    ran to a ceiling of 1.0 and produced optimistic projections. A straight
    line in log space assumes the current rate of improvement simply continues,
    which learning curves never do, so whatever it projects is an upper bound.
    That is the useful direction to be wrong in: if even this says more data
    will not help, more data will not help.
    """
    sizes, scores = np.asarray(sizes, float), np.asarray(scores, float)
    x = np.log2(sizes)
    slope, intercept = np.polyfit(x, scores, 1)
    pred = slope * x + intercept
    ss_res = float(((scores - pred) ** 2).sum())
    ss_tot = float(((scores - scores.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (lambda n: float(slope * np.log2(n) + intercept)), slope, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_mpnet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--weight", type=int, default=100,
                    help="CE upweighting factor, the modal choice from the "
                         "evaluation run; held fixed so only gold size varies")
    ap.add_argument("--sizes", default="50,100,150,200,0",
                    help="gold training sizes; 0 means the full training fold")
    args = ap.parse_args()
    sfx = args.suffix
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mapping = load_mapping(VARIANT)
    proj = pd.read_csv(PROJECTS, low_memory=False)
    gold = proj[proj.research_subjects.notna()].copy()
    gold["primary_field"] = gold.research_subjects.map(lambda s: primary_field(s, mapping))
    gold = gold[gold.primary_field.notna()].reset_index(drop=True)
    vc = gold.primary_field.value_counts()
    gold = gold[~gold.primary_field.isin(vc[vc < 5].index)].reset_index(drop=True)

    folds = json.load(open(DIR / f"folds_{VARIANT}.json"))["folds"][: args.folds]
    if set(folds[0]["train"]) | set(folds[0]["test"]) != set(gold.project_id):
        sys.exit("gold rebuild does not match the frozen folds; something upstream changed")

    classes = sorted(gold.primary_field.unique())
    y = gold.primary_field.values
    loc = {p: i for i, p in enumerate(gold.project_id)}
    gold_text = (gold.title_clean.fillna("") + ". "
                 + gold.abstract_text_clean.fillna("")).str.strip().values

    pemb = np.load(DIR / f"project_embeddings{sfx}.npy")
    pidx = pd.read_csv(DIR / "project_embedding_index.csv")
    if len(pidx) != len(pemb):
        sys.exit(f"embedding index has {len(pidx)} rows against {len(pemb)} vectors. "
                 f"Stale index; re-run embed_texts_mpnet.py --projects-only.")
    prow = {p: i for i, p in enumerate(pidx.project_id)}
    Xg = np.stack([pemb[prow[p]] for p in gold.project_id])

    corpus = pd.read_csv(CORPUS, low_memory=False)
    corpus["primary_field"] = corpus.research_subjects.map(lambda s: primary_field(s, mapping))
    cemb = np.load(DIR / f"corpus_embeddings{sfx}.npy")
    cidx = pd.read_csv(DIR / "corpus_embedding_index.csv")
    corpus = corpus.set_index("project_id").reindex(cidx.project_id).reset_index()
    overlap = corpus.project_id.isin(set(proj.project_id)).values
    ckeep = corpus.primary_field.isin(classes).values & ~overlap
    Xc, yc = cemb[ckeep], corpus.primary_field.values[ckeep]
    ctext = (corpus.title.fillna("") + ". "
             + corpus.abstract_text.fillna("")).str.strip().values[ckeep]
    print(f"gold {len(gold)}, {len(classes)} classes | corpus {ckeep.sum()} "
          f"(leakage guard dropped {int(overlap.sum())})")

    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                          stop_words="english", max_features=200_000)
    Zc = vec.fit_transform(ctext)
    Zg = vec.transform(gold_text)
    print(f"TF-IDF {Zc.shape[1]} features [{time.time() - t0:.0f}s]")

    n_train = min(len(f["train"]) for f in folds)
    sizes = [n_train if s == 0 else s for s in
             (int(x) for x in args.sizes.split(","))]
    sizes = sorted({s for s in sizes if 10 <= s <= n_train})
    print(f"sizes {sizes} (full training fold is {n_train})\n")

    rng = np.random.default_rng(SEED)
    rows = []
    for size in sizes:
        for fold in folds:
            tr_all = np.array([loc[p] for p in fold["train"]])
            te = [loc[p] for p in fold["test"]]
            tr = stratified_subset(y, tr_all, size, rng)
            w = np.concatenate([np.ones(len(yc)), np.full(len(tr), float(args.weight))])
            ytr = np.concatenate([yc, y[tr]])
            emb = LogisticRegression(max_iter=MAX_ITER, C=C_EMB, class_weight="balanced")
            emb.fit(np.vstack([Xc, Xg[tr]]), ytr, sample_weight=w)
            tf = LogisticRegression(max_iter=MAX_ITER, C=C_TFIDF, class_weight="balanced")
            tf.fit(spvstack([Zc, Zg[tr]]), ytr, sample_weight=w)
            proba = (emb.predict_proba(Xg[te]) + tf.predict_proba(Zg[te])) / 2
            pred = np.array(emb.classes_)[np.argmax(proba, axis=1)]
            rows.append(dict(size=size, n_actual=len(tr), split=fold["split"],
                             macro_f1=f1_score(y[te], pred, average="macro"),
                             accuracy=accuracy_score(y[te], pred)))
            print(f"  size {size:>4} split {fold['split']} "
                  f"F1 {rows[-1]['macro_f1']:.3f} acc {rows[-1]['accuracy']:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "gold_learning_curve.csv", index=False)
    summary = res.groupby("size")[["macro_f1", "accuracy"]].agg(["mean", "std"]).round(3)
    print("\n=== learning curve over gold set size ===")
    print(summary.to_string())

    means = res.groupby("size")[["macro_f1", "accuracy"]].mean()
    print("\n=== gain per step ===")
    prev = None
    for size, row in means.iterrows():
        if prev is not None:
            print(f"  {prev[0]:>4} -> {size:>4}: macro-F1 {row.macro_f1 - prev[1]:+.3f}, "
                  f"accuracy {row.accuracy - prev[2]:+.3f}")
        prev = (size, row.macro_f1, row.accuracy)

    project, slope, r2 = log_linear_projection(means.index.values, means.macro_f1.values)
    here = float(means.macro_f1.iloc[-1])
    n_here = int(means.index[-1])
    sd = float(res[res["size"] == n_here].macro_f1.std())
    print(f"\n=== projection ===")
    print(f"  observed macro-F1 {here:.3f} at {n_here} gold projects "
          f"(fold sd {sd:.3f})")
    print(f"  fitted gain per doubling of the gold set: {slope:+.3f} macro-F1 "
          f"(R2 {r2:.2f})")
    print("  this assumes the current rate continues, which learning curves do "
          "not,\n  so treat every figure below as an upper bound:")
    for n in (400, 500, 750, 1000, 2000):
        print(f"    at {n:>5}: {project(n):.3f}  ({project(n) - here:+.3f})")

    gain = project(2 * n_here) - here
    print("\n=== verdict ===")
    if gain >= 0.05:
        print(f"  Worth labelling. Doubling the gold set projects {gain:+.3f} macro-F1 "
              f"even before allowing for the curve flattening, which is well clear "
              f"of the {sd:.3f} fold-to-fold noise.")
    elif gain >= 0.02:
        print(f"  Marginal. Doubling projects {gain:+.3f} macro-F1 at best, against "
              f"fold-to-fold noise of {sd:.3f}. Only worth a day if you have one "
              f"spare, and the real gain will be smaller than this.")
    else:
        print(f"  Not worth labelling. Doubling the gold set projects {gain:+.3f} "
              f"macro-F1 at best, inside the {sd:.3f} fold-to-fold standard "
              f"deviation. The constraint is the difficulty of the task and the "
              f"noise in the funder labels, not the number of examples. Take "
              f"coarser classes or report the figure you have.")
    print(f"\nWrote gold_learning_curve.csv. Total {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
