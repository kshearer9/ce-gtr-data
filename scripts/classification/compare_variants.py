"""Compare the two crosswalk variants, James's and Kirsty's, end to end.

Both variants were run through the identical pipeline by run_variant.py: the
same 276 gold projects, the same wider GtR corpus, the same set-ups A to H, the
same 25 cross-validation splits. The only thing that differs is the mapping from
GtR research subject to OpenAlex field. That makes the two result sets paired,
so the comparison uses the Wilcoxon signed-rank test on matched splits rather
than an unpaired test on the two means (Demsar, 2006).

Three questions are answered here:

    1. Within each variant, does the ensemble (H) beat the weighted corpus
       set-up (G) and the projects-only baseline (A)?
    2. Between variants, does either crosswalk produce a better classifier?
    3. Where exactly do the two crosswalks disagree, and what does the
       disagreement cost on the outcome side?

Question 3 matters more than question 2. If the two schemes are statistically
indistinguishable, the choice cannot be made on performance and has to be made
on whether the scheme supports the input-output comparison the project is for.

Run from the repo root, after both variants of run_variant.py have completed:
    /opt/anaconda3/bin/python scripts/classification/compare_variants.py

Writes data/classification/results/variant_comparison.csv and prints the
tables reproduced in the methodology.
"""
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "classification"
RESULTS = DIR / "results"
VARIANTS = ("james", "kirsty")
PUBS = DIR / "publication_embedding_index.csv"


def load(variant: str) -> pd.DataFrame:
    path = RESULTS / f"setups_{variant}.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Run run_variant.py --crosswalk {variant} first.")
    return pd.read_csv(path).sort_values("split")


def paired(a: pd.Series, b: pd.Series) -> float:
    """Wilcoxon signed-rank p value on matched splits."""
    return float(wilcoxon(a.values, b.values).pvalue)


def main() -> None:
    setups = {v: load(v) for v in VARIANTS}
    rows = []

    print("=== 1. Within-variant set-up comparison (macro-F1) ===")
    for v, frame in setups.items():
        means = frame.groupby("setup")[["macro_f1", "accuracy"]].mean().round(3)
        print(f"\n[{v}]")
        print(means.to_string())
        for better, worse in (("H", "G"), ("H", "A"), ("G", "A")):
            x = frame.loc[frame.setup == better, "macro_f1"]
            y = frame.loc[frame.setup == worse, "macro_f1"]
            p = paired(x, y)
            print(f"  {better} vs {worse}: {x.mean() - y.mean():+.3f} macro-F1, p = {p:.4g}")
            rows.append(dict(test="within", variant=v, comparison=f"{better} vs {worse}",
                             delta_macro_f1=round(x.mean() - y.mean(), 4), p_value=round(p, 5)))

    print("\n=== 2. Between-variant comparison, same splits ===")
    for setup in ("A", "G", "H"):
        j = setups["james"].loc[setups["james"].setup == setup]
        k = setups["kirsty"].loc[setups["kirsty"].setup == setup]
        for metric in ("macro_f1", "accuracy"):
            p = paired(j[metric], k[metric])
            print(f"  {setup} {metric:9s}: james {j[metric].mean():.3f} "
                  f"vs kirsty {k[metric].mean():.3f}, p = {p:.3f}")
            rows.append(dict(test="between", variant="james vs kirsty",
                             comparison=f"{setup} {metric}",
                             delta_macro_f1=round(j[metric].mean() - k[metric].mean(), 4),
                             p_value=round(p, 5)))

    print("\n=== 3. Where the crosswalks disagree ===")
    gj = pd.read_csv(DIR / "gold_james.csv")[["project_id", "primary_field"]]
    gk = pd.read_csv(DIR / "gold_kirsty.csv")[["project_id", "primary_field"]]
    merged = gj.merge(gk, on="project_id", suffixes=("_j", "_k"))
    agree = (merged.primary_field_j == merged.primary_field_k)
    print(f"  Same label on {agree.sum()}/{len(merged)} gold projects ({agree.mean():.1%})")
    moves = (merged.loc[~agree]
             .groupby(["primary_field_j", "primary_field_k"]).size()
             .sort_values(ascending=False)
             .rename("projects"))
    print(moves.to_string())

    only_j = sorted(set(gj.primary_field) - set(gk.primary_field))
    only_k = sorted(set(gk.primary_field) - set(gj.primary_field))
    print(f"\n  Classes only in james: {only_j or 'none'}")
    print(f"  Classes only in kirsty: {only_k or 'none'}")

    if PUBS.exists():
        pubs = pd.read_csv(PUBS)
        print("\n  Outcome-side cost: publications sitting in a field that the "
              "input side cannot express")
        for field in only_j:
            n = int((pubs.field == field).sum())
            print(f"    {field}: {n} publications unreachable under the kirsty scheme")
            rows.append(dict(test="coverage", variant="kirsty", comparison=field,
                             delta_macro_f1=None, p_value=None))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "variant_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
