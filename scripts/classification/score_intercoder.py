"""Three-way agreement: two human coders and the model, on the same projects.

The verification told you the model agrees with you 69.8% of the time on tier 2.
On its own that number floats: a reader has no way to know whether 70% is poor
or close to the limit of what the task allows. A second coder working the same
projects blind supplies the missing reference point.

Three comparisons come out of it, and the first is the one that matters:

    coder A vs coder B   how far two careful humans agree on this scheme. This
                         is the ceiling. If it is 75%, then a model at 70% is
                         doing most of what is achievable, and the story is
                         about the difficulty of assigning one discipline to
                         interdisciplinary work rather than about a weak model.
    model vs coder A     already measured
    model vs coder B     the same test against a coder who did not build the
                         crosswalk, so it is not contaminated by familiarity
                         with the scheme

Reported as raw agreement and as Krippendorff's alpha. Alpha is the measure
content analysis conventionally uses because it corrects for the agreement two
coders would reach by chance on an imbalanced distribution, which raw agreement
does not; with Engineering at roughly a third of your data, raw agreement
flatters everybody. The conventional bar is alpha of 0.80, with 0.667 sometimes
accepted for tentative conclusions provided the shortfall is discussed.
Krippendorff is the source to cite for that, not this docstring, and you should
read it before it goes in your text.

Note what alpha does and does not apply to. It is a reliability statistic for a
CODING SCHEME, so coder A against coder B is its proper use. Applying it to the
model is a looser analogy: a classifier is not a coder and there is no
convention that says a model must clear 0.80. Report the model comparisons as
agreement with a human reference and interpret them against the ceiling.

    python scripts/classification/score_intercoder.py \\
        --second data/validation/discipline_coding_SECOND_CODER.xlsx

Partial completion is fine. Only rows both coders filled in are compared, and
the count is printed so you can see what the figure rests on.
"""
from pathlib import Path
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "validation"
KEY = VAL / "discipline_verification_KEY.csv"
FIRST = VAL / "verification_final_coding.csv"
UNCLEAR = "UNCLEAR"

# The first coding pass was done on the twelve-class scheme, before the merge.
# The second coder's dropdown already holds the ten. Both are mapped through so
# every comparison sits in one taxonomy; without this the first coder is scored
# against a scheme that no longer exists and every figure comes out too low.
FIELD_MERGE = {
    "Biochemistry, Genetics and Molecular Biology": "Agricultural and Biological Sciences",
    "Materials Science": "Engineering",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def krippendorff_alpha_nominal(units):
    """Alpha for nominal data.

    units: list of lists, one per item, holding that item's codes from whichever
    coders rated it. Items with fewer than two codes contribute nothing, which
    is the standard treatment of missing values rather than a shortcut.
    """
    usable = [u for u in units if len(u) >= 2]
    if not usable:
        return float("nan")
    values = sorted({v for u in usable for v in u})
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    coincidence = np.zeros((k, k))
    for u in usable:
        m = len(u)
        for a, b in itertools.permutations(u, 2):
            coincidence[idx[a], idx[b]] += 1.0 / (m - 1)
    n_total = coincidence.sum()
    if n_total == 0:
        return float("nan")
    marginals = coincidence.sum(axis=1)
    observed = np.trace(coincidence)
    expected = float((marginals * (marginals - 1)).sum() / (n_total - 1))
    do = 1 - observed / n_total
    de = 1 - expected / n_total
    return float(1 - do / de) if de > 0 else float("nan")


def pair(frame, a, b, label):
    both = frame[frame[a].notna() & frame[b].notna()]
    both = both[(both[a].str.upper() != UNCLEAR) & (both[b].str.upper() != UNCLEAR)]
    n = len(both)
    if n == 0:
        print(f"  {label:28s} no overlapping rows")
        return None
    hits = int((both[a] == both[b]).sum())
    lo, hi = wilson(hits, n)
    alpha = krippendorff_alpha_nominal([[r[a], r[b]] for _, r in both.iterrows()])
    print(f"  {label:28s} n={n:3d}  agreement {hits/n:.1%} [{lo:.1%}, {hi:.1%}]  "
          f"alpha {alpha:.3f}")
    return dict(comparison=label, n=n, agreement=round(hits / n, 4),
                ci_low=round(lo, 4), ci_high=round(hi, 4), alpha=round(alpha, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--second", required=True, help="the second coder's sheet")
    ap.add_argument("--sheet", default="Coding")
    args = ap.parse_args()

    for p in (KEY, FIRST):
        if not p.exists():
            sys.exit(f"Missing {p}.")
    second_path = Path(args.second)
    if not second_path.exists():
        sys.exit(f"Missing {second_path}.")

    key = pd.read_csv(KEY)
    first = pd.read_csv(FIRST)
    second = pd.read_excel(second_path, sheet_name=args.sheet)

    df = key[["sample_id", "project_id", "tier", "model_field"]].copy()
    df = df.merge(first[["sample_id", "final_field"]].rename(
        columns={"final_field": "coder_a"}), on="sample_id", how="left")
    df = df.merge(second[["sample_id", "your_field"]].rename(
        columns={"your_field": "coder_b"}), on="sample_id", how="left")
    for c in ("coder_a", "coder_b", "model_field"):
        df[c] = df[c].astype("string").str.strip().replace(FIELD_MERGE)

    done = int(df.coder_b.notna().sum())
    print(f"second coder has completed {done} of {len(df)} rows\n")
    if done == 0:
        sys.exit("Nothing to compare yet.")

    print("=== all coded rows ===")
    rows = [pair(df, "coder_a", "coder_b", "coder A vs coder B"),
            pair(df, "model_field", "coder_a", "model vs coder A"),
            pair(df, "model_field", "coder_b", "model vs coder B")]

    for t in sorted(df.tier.dropna().unique()):
        sub = df[df.tier == t]
        if sub.coder_b.notna().sum() >= 10:
            print(f"\n=== tier {int(t)} ===")
            for a, b, lab in (("coder_a", "coder_b", "coder A vs coder B"),
                              ("model_field", "coder_a", "model vs coder A"),
                              ("model_field", "coder_b", "model vs coder B")):
                r = pair(sub, a, b, lab)
                if r:
                    r["tier"] = int(t)
                    rows.append(r)

    rows = [r for r in rows if r]
    human = next((r for r in rows if r["comparison"] == "coder A vs coder B"
                  and "tier" not in r), None)
    model = next((r for r in rows if r["comparison"] == "model vs coder A"
                  and "tier" not in r), None)
    if human and model:
        print("\n=== interpretation ===")
        print(f"  two humans agree {human['agreement']:.1%}; the model agrees with "
              f"coder A {model['agreement']:.1%}")
        gap = human["agreement"] - model["agreement"]
        if gap <= 0:
            print("  The model is at or above the human ceiling. Say so plainly, "
                  "and note\n  that it means the scheme's ambiguity, not the "
                  "model, is the binding constraint.")
        elif gap <= 0.10:
            print(f"  The model sits {gap:.1%} below the human ceiling. That is the "
                  f"headline\n  finding: most of the error is the task, not the "
                  f"classifier.")
        else:
            print(f"  The model sits {gap:.1%} below the human ceiling, so a real "
                  f"share of the\n  error is the model rather than the task. "
                  f"Report both figures.")
        if not np.isnan(human["alpha"]):
            a = human["alpha"]
            verdict = ("clears the conventional 0.80 bar" if a >= 0.80 else
                       "sits between 0.667 and 0.80, acceptable for tentative "
                       "conclusions if the shortfall is discussed" if a >= 0.667
                       else "falls below 0.667, so the coding scheme itself is "
                            "not reliably applied and that is a finding about "
                            "the scheme")
            print(f"  Human alpha of {a:.3f} {verdict}.")

    out = ROOT / "data" / "classification" / "results" / "intercoder_agreement.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
