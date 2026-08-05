"""Training-corpus comparison for the discipline classifier (set-ups A to G).

Extends the three-corpus experiment suggested by the project's text-mining
adviser to seven, all evaluated on the same frozen folds so the comparison is
fair:

    A  CE project abstracts only
    B  publication abstracts only
    C  projects + publications
    D  wider GtR corpus only
    E  corpus + projects, unweighted
    F  corpus subsampled to the CE class distribution
    G  corpus + projects, CE instances upweighted (Jiang and Zhai, 2007)

For set-up G the weighting factor is chosen by NESTED cross-validation: the
factor is selected on inner folds drawn from the training portion, and scored
on the outer fold, which the selection never sees. Selecting it on the outer
folds directly would bias the reported figure upwards (Cawley and Talbot, 2010).

Set-ups B and C apply a leakage guard: any publication belonging to a project
in the current test fold is removed from training, since classifying a project
using a model trained on its own outputs would inflate the result.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/run_corpus_setups.py
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
RESULTS = DIR / "results"
GOLD = DIR / "gold_labelled_projects.csv"
FOLDS = DIR / "cv_folds_FROZEN.json"

C_VAL, MAX_ITER = 3.0, 3000
WEIGHT_GRID = [1, 5, 10, 25, 50, 100]
SEED = 20260803


def fit(X, y, weights=None):
    clf = LogisticRegression(max_iter=MAX_ITER, C=C_VAL, class_weight="balanced")
    clf.fit(X, y, sample_weight=weights)
    return clf


def score(y_true, pred):
    return dict(macro_f1=f1_score(y_true, pred, average="macro"),
                accuracy=accuracy_score(y_true, pred))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help='embedding filename suffix, e.g. "_mpnet"')
    sfx = ap.parse_args().suffix
    if sfx:
        print(f"Using embeddings with suffix {sfx!r}")
    for path in (GOLD, FOLDS):
        if not path.exists():
            sys.exit(f"Missing input: {path}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(GOLD)
    folds = json.load(open(FOLDS))["folds"]
    classes = sorted(gold.primary_field.unique())
    y = gold.primary_field.values
    loc = {p: i for i, p in enumerate(gold.project_id)}

    pemb = np.load(DIR / f"project_embeddings{sfx}.npy")
    pidx = pd.read_csv(DIR / "project_embedding_index.csv")
    prow = {p: i for i, p in enumerate(pidx.project_id)}
    X = np.stack([pemb[prow[p]] for p in gold.project_id])

    bemb = np.load(DIR / f"publication_embeddings{sfx}.npy")
    bidx = pd.read_csv(DIR / "publication_embedding_index.csv")
    bkeep = bidx.field.isin(classes).values
    Xb, yb, bproj = bemb[bkeep], bidx.field.values[bkeep], bidx.project_id.values[bkeep]

    cemb = np.load(DIR / f"corpus_embeddings{sfx}.npy")
    cidx = pd.read_csv(DIR / "corpus_embedding_index.csv")
    ckeep = cidx.primary_field.isin(classes).values
    Xc, yc = cemb[ckeep], cidx.primary_field.values[ckeep]

    print(f"CE projects {len(gold)} | publications in scope {bkeep.sum()} "
          f"| corpus in scope {ckeep.sum()}\n")

    rng = np.random.default_rng(SEED)
    target = gold.primary_field.value_counts(normalize=True)
    matched = np.concatenate([
        rng.choice(np.flatnonzero(yc == cls),
                   size=min((yc == cls).sum(), max(1, int(round(prop * len(yc))))),
                   replace=False)
        for cls, prop in target.items()])

    rows, chosen = [], []
    for fold in folds:
        tr = [loc[p] for p in fold["train"]]
        te = [loc[p] for p in fold["test"]]
        guard = ~np.isin(bproj, fold["test"])          # leakage guard
        y_te = y[te]

        builds = {
            "A": lambda: (X[tr], y[tr], None),
            "B": lambda: (Xb[guard], yb[guard], None),
            "C": lambda: (np.vstack([X[tr], Xb[guard]]),
                          np.concatenate([y[tr], yb[guard]]), None),
            "D": lambda: (Xc, yc, None),
            "E": lambda: (np.vstack([Xc, X[tr]]), np.concatenate([yc, y[tr]]), None),
            "F": lambda: (Xc[matched], yc[matched], None),
        }
        for name, build in builds.items():
            Xtr, ytr, w = build()
            rows.append(dict(setup=name, split=fold["split"],
                             **score(y_te, fit(Xtr, ytr, w).predict(X[te]))))

        # --- G: weight chosen on inner folds only ---
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        best, best_score = WEIGHT_GRID[0], -1.0
        for mult in WEIGHT_GRID:
            inner_scores = []
            for itr, ite in inner.split(X[tr], y[tr]):
                sub = np.asarray(tr)[itr]
                Xtr = np.vstack([Xc, X[sub]])
                ytr = np.concatenate([yc, y[sub]])
                w = np.concatenate([np.ones(len(yc)), np.full(len(sub), float(mult))])
                clf = fit(Xtr, ytr, w)
                held = np.asarray(tr)[ite]
                inner_scores.append(f1_score(y[held], clf.predict(X[held]),
                                             average="macro"))
            mean_inner = float(np.mean(inner_scores))
            if mean_inner > best_score:
                best, best_score = mult, mean_inner
        chosen.append(best)
        Xtr = np.vstack([Xc, X[tr]])
        ytr = np.concatenate([yc, y[tr]])
        w = np.concatenate([np.ones(len(yc)), np.full(len(tr), float(best))])
        rows.append(dict(setup="G", split=fold["split"],
                         **score(y_te, fit(Xtr, ytr, w).predict(X[te]))))
        print(f"  split {fold['split']:2d} done (G weight x{best})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"corpus_setups{sfx}.csv", index=False)
    summary = res.groupby("setup")[["macro_f1", "accuracy"]].agg(["mean", "std"]).round(3)
    summary.to_csv(RESULTS / f"corpus_setups_summary{sfx}.csv")
    print("\n=== Training-corpus comparison ===")
    print(summary.to_string())
    print(f"\nG weights selected across outer folds: "
          f"{pd.Series(chosen).value_counts().to_dict()}")
    print(f"Wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
