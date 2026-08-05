"""Method bake-off for the discipline classifier.

Compares candidate classification methods on identical cross-validation folds,
so any difference between them is attributable to the method rather than to the
split. Folds are read from cv_folds_FROZEN.json and never regenerated here.

Macro-averaged F1 is the primary metric, declared before any method was run, so
that small fields are not sacrificed to headline accuracy (Forman and Scholz,
2010). Accuracy and top-2 accuracy are reported alongside; top-2 is informative
because 60% of the labelled projects carry more than one field.

Methods
    M1   TF-IDF + logistic regression      classical baseline
    M1b  TF-IDF + multinomial Naive Bayes  taught baseline (COMP42415)
    M2   SBERT embeddings + logistic regression
    M3   nearest class centroid over SBERT embeddings
    M6   soft-vote ensemble of M1 and M2

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/run_bakeoff.py

Outputs one CSV per method plus a summary table in data/classification/results/.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
RESULTS = DIR / "results"
GOLD = DIR / "gold_labelled_projects.csv"
FOLDS = DIR / "cv_folds_FROZEN.json"

C_TFIDF, C_EMB, MAX_ITER = 1.0, 3.0, 3000


def metrics(y_true, pred, proba, classes):
    top2 = np.argsort(-proba, axis=1)[:, :2]
    return dict(
        macro_f1=f1_score(y_true, pred, average="macro"),
        micro_f1=f1_score(y_true, pred, average="micro"),
        weighted_f1=f1_score(y_true, pred, average="weighted"),
        accuracy=accuracy_score(y_true, pred),
        top2_accuracy=np.mean([y_true[i] in classes[top2[i]] for i in range(len(y_true))]),
    )


def tfidf_pipe(classifier):
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                        stop_words="english"),
        classifier,
    )


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
    gold["text"] = (gold.title_clean.fillna("") + ". "
                    + gold.abstract_text_clean.fillna("")).str.strip()
    folds = json.load(open(FOLDS))["folds"]
    y = gold.primary_field.values
    loc = {p: i for i, p in enumerate(gold.project_id)}

    emb = np.load(DIR / f"project_embeddings{sfx}.npy")
    index = pd.read_csv(DIR / "project_embedding_index.csv")
    row_of = {p: i for i, p in enumerate(index.project_id)}
    X = np.stack([emb[row_of[p]] for p in gold.project_id])

    print(f"{len(gold)} projects, {gold.primary_field.nunique()} classes, "
          f"{len(folds)} splits\n")

    def run(name, predict):
        rows = []
        for fold in folds:
            tr = [loc[p] for p in fold["train"]]
            te = [loc[p] for p in fold["test"]]
            pred, proba, classes = predict(tr, te)
            rows.append(dict(split=fold["split"], **metrics(y[te], pred, proba, classes)))
        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / f"{name}{sfx}.csv", index=False)
        print(f"{name:16s} macro-F1 {frame.macro_f1.mean():.3f} "
              f"+/- {frame.macro_f1.std():.3f} | accuracy {frame.accuracy.mean():.3f} "
              f"| top-2 {frame.top2_accuracy.mean():.3f}", flush=True)
        return frame.assign(method=name)

    def m1(tr, te):
        pipe = tfidf_pipe(LogisticRegression(max_iter=MAX_ITER, C=C_TFIDF,
                                             class_weight="balanced"))
        pipe.fit(gold.text.iloc[tr], y[tr])
        return (pipe.predict(gold.text.iloc[te]),
                pipe.predict_proba(gold.text.iloc[te]), pipe.classes_)

    def m1b(tr, te):
        pipe = tfidf_pipe(MultinomialNB())
        pipe.fit(gold.text.iloc[tr], y[tr])
        return (pipe.predict(gold.text.iloc[te]),
                pipe.predict_proba(gold.text.iloc[te]), pipe.classes_)

    def m2(tr, te):
        clf = LogisticRegression(max_iter=MAX_ITER, C=C_EMB,
                                 class_weight="balanced").fit(X[tr], y[tr])
        return clf.predict(X[te]), clf.predict_proba(X[te]), clf.classes_

    def m3(tr, te):
        classes = np.unique(y[tr])
        cent = np.stack([X[tr][y[tr] == c].mean(axis=0) for c in classes])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        sims = X[te] @ cent.T
        proba = np.exp(sims * 10)
        proba /= proba.sum(axis=1, keepdims=True)
        return classes[np.argmax(sims, axis=1)], proba, classes

    def m6(tr, te):
        pipe = tfidf_pipe(LogisticRegression(max_iter=MAX_ITER, C=C_TFIDF,
                                             class_weight="balanced"))
        pipe.fit(gold.text.iloc[tr], y[tr])
        clf = LogisticRegression(max_iter=MAX_ITER, C=C_EMB,
                                 class_weight="balanced").fit(X[tr], y[tr])
        assert list(pipe.classes_) == list(clf.classes_)
        proba = (pipe.predict_proba(gold.text.iloc[te]) + clf.predict_proba(X[te])) / 2
        return pipe.classes_[np.argmax(proba, axis=1)], proba, pipe.classes_

    frames = [run("M1_tfidf_lr", m1), run("M1b_tfidf_nb", m1b), run("M2_sbert_lr", m2),
              run("M3_centroid", m3), run("M6_softvote", m6)]

    summary = (pd.concat(frames)
               .groupby("method")[["macro_f1", "accuracy", "top2_accuracy"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(RESULTS / f"bakeoff_summary{sfx}.csv")
    print(f"\nWrote per-method CSVs and bakeoff_summary.csv to {RESULTS}")


if __name__ == "__main__":
    main()
