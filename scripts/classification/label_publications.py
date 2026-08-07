"""Assign an OpenAlex field to every linked publication, from three sources.

The output side of the input-output comparison needs each publication expressed
in the same 26-field taxonomy as the projects. The three bibliometric sources
arrive in three different states:

    OpenAlex  already carries domain/field/subfield natively. Authoritative.
    Scopus    carries ASJC subject areas, a finer level of the same lineage.
    WoS       carries its own categories and Citation Topics, a separate scheme.

Rather than hand-build two more crosswalks, the correspondences are LEARNED from
the papers that appear in more than one source. A DOI identifies one article, so
a paper held by both Scopus and OpenAlex gives a directly observed pairing of a
Scopus subject area with an OpenAlex field. Thousands of such pairings make the
mapping a measurement rather than a judgement, and it comes with its own error
rate attached.

Four stages, in descending order of measured accuracy:

    1. DIRECT      the paper is in OpenAlex; take its field. No inference.
    2. WOS TOPIC   map its Citation Topic (micro level) through the derived table.
    3. CLASSIFIER  predict from title and abstract, trained on the OpenAlex-
                   labelled publications.
    4. SCOPUS      map its ASJC subject areas through the derived table.

The ordering is empirical, not assumed, and it inverts what the shared ASJC
lineage would suggest. Scopus subject areas are assigned at JOURNAL level, so a
paper inherits every area its journal carries, a median of three, and journal
scope is a poor guide to an individual article's discipline: the route scores
45%. WoS Citation Topics are assigned per ARTICLE by citation clustering and
transfer at 79%. A classifier over the abstract sits between them at 65%. Scopus
is therefore demoted to a last resort for papers with no abstract and no
Citation Topic.

Each derived table is validated by splitting the overlap in half, deriving on one
half and scoring on the other, so the reported accuracy is out-of-sample rather
than the training fit. Labels below a support threshold are refused rather than
guessed, in keeping with the reject option applied on the project side.

Run from the repo root:
    /opt/anaconda3/bin/python scripts/classification/label_publications.py

Writes data/cleaned/outcomes/publications_labelled.csv, one row per unique
publication, with the field, the route that produced it, and the measured
accuracy of that route.
"""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "cleaned" / "outcomes"
OPENALEX = OUT_DIR / "openalex_all_outcomes_clean.csv"
SCOPUS = OUT_DIR / "scopus_all_outcomes_clean.csv"
WOS = OUT_DIR / "wos_all_outcomes_clean.csv"
OUT = OUT_DIR / "publications_labelled.csv"
MAPPINGS = OUT_DIR / "derived_field_mappings.csv"

MIN_SUPPORT = 5       # observed pairings needed before a label is trusted
MIN_SHARE = 0.50      # the modal field must hold at least this share
SEED = 20260806


def norm_doi(value) -> str | None:
    """Normalise a DOI so the three sources can be joined on it."""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return s if s.startswith("10.") else None


def split_labels(value, sep=r"[;|]"):
    if pd.isna(value):
        return []
    return [p.strip() for p in re.split(sep, str(value)) if p.strip()]


def derive_mapping(pairs: pd.DataFrame, name: str) -> tuple[dict, float, int]:
    """Learn label -> OpenAlex field from observed pairings.

    Held-out validation: derive on a random half, score on the other, so the
    accuracy reported is out-of-sample. The returned table is then rebuilt on
    all the data, since more evidence is strictly better once the method is
    validated.
    """
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(pairs)) < 0.5
    train, test = pairs[mask], pairs[~mask]

    def build(frame):
        g = frame.groupby("label").oa_field.agg(
            n="size", top=lambda s: s.mode().iat[0],
            share=lambda s: s.value_counts(normalize=True).iat[0])
        g = g[(g.n >= MIN_SUPPORT) & (g.share >= MIN_SHARE)]
        return dict(zip(g.index, g.top))

    trial = build(train)
    scored = test[test.label.isin(trial)]
    acc = float((scored.label.map(trial) == scored.oa_field).mean()) if len(scored) else 0.0

    full = build(pairs)
    print(f"  {name}: {len(full)} labels retained of {pairs.label.nunique()} seen "
          f"| held-out accuracy {acc:.1%} on {len(scored)} papers")
    return full, acc, len(scored)


def main() -> None:
    for path in (OPENALEX, SCOPUS, WOS):
        if not path.exists():
            sys.exit(f"Missing {path}. Run the collection and cleaning first.")

    oa = pd.read_csv(OPENALEX, low_memory=False)
    sc = pd.read_csv(SCOPUS, low_memory=False)
    wo = pd.read_csv(WOS, low_memory=False)
    for frame in (oa, sc, wo):
        frame["doi_norm"] = frame.doi.map(norm_doi)

    # OpenAlex is the authority: one field per DOI.
    authority = (oa.dropna(subset=["doi_norm", "field"])
                   .drop_duplicates("doi_norm")
                   .set_index("doi_norm").field)
    print(f"OpenAlex authority: {len(authority)} papers with a field\n")

    print("Deriving mappings from cross-source overlaps")

    # --- Scopus ASJC subject areas ------------------------------------------
    sc_pairs = []
    for _, r in sc.dropna(subset=["doi_norm", "subject_areas"]).iterrows():
        if r.doi_norm in authority.index:
            for lab in split_labels(r.subject_areas):
                sc_pairs.append((lab, authority[r.doi_norm]))
    sc_pairs = pd.DataFrame(sc_pairs, columns=["label", "oa_field"])
    sc_map, sc_acc, sc_n = derive_mapping(sc_pairs, "Scopus ASJC")

    # --- WoS Citation Topics, micro level -----------------------------------
    wo_pairs = []
    for _, r in wo.dropna(subset=["doi_norm", "citation_topic_micro"]).iterrows():
        if r.doi_norm in authority.index:
            for lab in split_labels(r.citation_topic_micro):
                wo_pairs.append((lab, authority[r.doi_norm]))
    wo_pairs = pd.DataFrame(wo_pairs, columns=["label", "oa_field"])
    wo_map, wo_acc, wo_n = derive_mapping(wo_pairs, "WoS Citation Topic")

    # --- A classifier over the abstract, trained on the OpenAlex labels ------
    # Sits between the two derived tables in accuracy and reaches papers that
    # have neither a Citation Topic nor a usable subject area, provided they
    # have an abstract.
    clf, clf_acc = None, 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline

        train = oa.dropna(subset=["field"]).drop_duplicates("doi_norm").copy()
        train["text"] = (train.title_clean.fillna("") + ". "
                         + train.abstract_clean.fillna("")).str.strip()
        train = train[train.text.str.len() > 40]
        counts = train.field.value_counts()
        train = train[train.field.isin(counts[counts >= 15].index)]

        def build_pipe():
            return make_pipeline(
                TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                stop_words="english"),
                LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced"))

        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        # n_jobs=1 deliberately: the corpus is small enough that parallelism
        # buys nothing, and the worker pool emits ResourceTracker noise on
        # teardown under Python 3.13 on macOS that looks like a failure.
        pred = cross_val_predict(build_pipe(), train.text.values, train.field.values,
                                 cv=cv, n_jobs=1)
        clf_acc = float(accuracy_score(train.field.values, pred))
        clf = build_pipe().fit(train.text.values, train.field.values)
        print(f"  Abstract classifier: trained on {len(train)} papers, "
              f"{train.field.nunique()} classes | cross-validated accuracy {clf_acc:.1%}")
    except ImportError:
        print("  Abstract classifier: scikit-learn unavailable, route skipped")

    # --- Assign, best measured route first -----------------------------------
    def vote(labels, mapping):
        """Modal field across a paper's labels, ignoring unmapped ones."""
        fields = [mapping[l] for l in labels if l in mapping]
        return pd.Series(fields).mode().iat[0] if fields else None

    def text_of(row):
        t = f"{row.get('title_clean') or ''}. {row.get('abstract_clean') or ''}".strip()
        return t if len(t) > 40 else None

    rows = {}
    for _, r in oa.dropna(subset=["doi_norm"]).iterrows():
        if pd.notna(r.get("field")):
            rows[r.doi_norm] = (r.field, "direct_openalex", 1.0, r.get("title"))

    # 2. WoS Citation Topics, the strongest inferred route
    for _, r in wo.dropna(subset=["doi_norm"]).iterrows():
        if r.doi_norm in rows:
            continue
        f = vote(split_labels(r.citation_topic_micro), wo_map)
        if f:
            rows[r.doi_norm] = (f, "wos_citation_topic", round(wo_acc, 3), r.get("title"))

    # 3. The abstract classifier
    if clf is not None:
        pending = []
        for frame in (wo, sc):
            for _, r in frame.dropna(subset=["doi_norm"]).iterrows():
                if r.doi_norm in rows:
                    continue
                txt = text_of(r)
                if txt:
                    pending.append((r.doi_norm, txt, r.get("title")))
        seen = set()
        pending = [x for x in pending if not (x[0] in seen or seen.add(x[0]))]
        if pending:
            preds = clf.predict([x[1] for x in pending])
            for (doi, _, title), f in zip(pending, preds):
                rows[doi] = (f, "abstract_classifier", round(clf_acc, 3), title)

    # 4. Scopus ASJC, last resort: journal-level and the weakest route
    for _, r in sc.dropna(subset=["doi_norm"]).iterrows():
        if r.doi_norm in rows:
            continue
        f = vote(split_labels(r.subject_areas), sc_map)
        if f:
            rows[r.doi_norm] = (f, "scopus_asjc", round(sc_acc, 3), r.get("title"))

    labelled = pd.DataFrame(
        [(d, f, m, a, t) for d, (f, m, a, t) in rows.items()],
        columns=["doi", "field", "field_route", "route_accuracy", "title"])

    # Confidence tiers, mirroring the treatment of the project side: an observed
    # label, an inferred label from a route that clears the bar, or an inferred
    # label reported as low confidence.
    def tier(row):
        if row.field_route == "direct_openalex":
            return 1
        return 2 if row.route_accuracy >= 0.70 else 3
    labelled["tier"] = labelled.apply(tier, axis=1)

    # Every DOI seen anywhere, so the unlabelled remainder is explicit rather
    # than silently absent.
    all_dois = set(oa.doi_norm.dropna()) | set(sc.doi_norm.dropna()) | set(wo.doi_norm.dropna())
    missing = sorted(all_dois - set(labelled.doi))
    if missing:
        labelled = pd.concat([labelled, pd.DataFrame(
            {"doi": missing, "field": None, "field_route": "unlabelled",
             "route_accuracy": np.nan, "title": None, "tier": np.nan})],
            ignore_index=True)

    labelled.to_csv(OUT, index=False, encoding="utf-8")
    pd.concat([
        pd.DataFrame({"source": "scopus_asjc", "label": list(sc_map),
                      "openalex_field": list(sc_map.values())}),
        pd.DataFrame({"source": "wos_citation_topic", "label": list(wo_map),
                      "openalex_field": list(wo_map.values())}),
    ]).to_csv(MAPPINGS, index=False, encoding="utf-8")

    print(f"\nPublications: {len(all_dois)} distinct DOIs across the three sources")
    for route, n in labelled.field_route.value_counts().items():
        acc = labelled.loc[labelled.field_route == route, "route_accuracy"].iloc[0]
        acc_s = "" if pd.isna(acc) else f"  (route accuracy {acc:.0%})"
        print(f"  {route:22s} {n:>5} ({n/len(all_dois):>3.0%}){acc_s}")
    got = int(labelled.field.notna().sum())
    print(f"  {'LABELLED':22s} {got:>5} ({got/len(all_dois):>3.0%})")
    print("\n  by confidence tier:")
    for t_, n in labelled.tier.value_counts().sort_index().items():
        print(f"    tier {int(t_)}: {n}")
    print(f"\nWrote {OUT}")
    print(f"Wrote {MAPPINGS}")


if __name__ == "__main__":
    main()
