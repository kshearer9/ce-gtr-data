"""Score the hand-coded screening validation sample against the rule's decisions.

`collect_gtr_projects.py` writes a validation template with an empty
`is_ce_manual` column. Nothing read it back, so the screening figures quoted in
the methodology were the only numbers in the chapter without a committed
source. This script closes that gap.

    python scripts/collection/score_screening.py

The sample was drawn stratified on the rule's own decision, roughly half from
the projects it admitted and half from those it rejected, because the retained
population is about 1.5 per cent of candidates and a proportional draw would
have contained too few admitted projects for filter errors to be visible.

That design decides which statistics mean anything:

    precision   of the projects the rule ADMITS, how many are genuinely
                circular economy. Conditions on the rule's positive decision,
                and that stratum was a random draw from every project the rule
                admits, so the estimate carries to the full dataset.

    NPV         the mirror figure for the rejected stratum, on the same
                argument.

    agreement   raw agreement and Cohen's kappa are computed and printed, but
    and kappa   they are NOT reportable as properties of the rule. Both depend
                on the balance of positives and negatives, and that balance was
                chosen when the sample was drawn rather than observed. Holding
                the rule's behaviour fixed and varying only the draw ratio moves
                kappa from roughly 0.60 to 0.84 to 0.71. They describe the
                evaluation sample and nothing beyond it.

Wilson intervals rather than the normal approximation, which at 50/53 returns
an upper bound above 1 and so cannot be right for a proportion. Matches the
convention in scripts/classification/score_verification.py.

One project appears twice in the sheet, so 100 coded rows cover 99 distinct
projects. Both figures are reported. The duplicate is a true positive, so
deduplicating moves precision by about a thousandth and leaves NPV untouched.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "validation"
SHEET = VAL / "screening_validation_100.xlsx"

RULE_COL = "filter_decision"
HUMAN_COL = "is_ce_manual"
KEEP, DROP, UNSURE = "keep", "drop", "unsure"
Z = 1.96


def wilson(k, n, z=Z):
    """Wilson score interval. Cannot leave [0, 1], unlike the Wald interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_sheet(path):
    """The sheet carries coding instructions above the header row, so find it."""
    if not path.exists():
        sys.exit(f"Coding sheet not found: {path}")
    raw = pd.read_excel(path, header=None)
    try:
        header = next(
            i for i in range(len(raw))
            if "project_id" in [str(x).strip() for x in raw.iloc[i].tolist()]
        )
    except StopIteration:
        sys.exit(f"No 'project_id' header row found in {path.name}")
    df = pd.read_excel(path, header=header)
    df = df.dropna(subset=["project_id"])
    for col in (RULE_COL, HUMAN_COL):
        if col not in df.columns:
            sys.exit(f"Column {col!r} not found in {path.name}")
    return df


def confusion(df):
    rule = df[RULE_COL].astype(str).str.strip().str.lower()
    human = df[HUMAN_COL].astype(str).str.strip().str.lower()

    unsure = int((human == UNSURE).sum())
    scorable = human.isin([KEEP, DROP]) & rule.isin([KEEP, DROP])
    rule, human = rule[scorable], human[scorable]

    return dict(
        TP=int(((rule == KEEP) & (human == KEEP)).sum()),
        FP=int(((rule == KEEP) & (human == DROP)).sum()),
        FN=int(((rule == DROP) & (human == KEEP)).sum()),
        TN=int(((rule == DROP) & (human == DROP)).sum()),
        unsure=unsure,
    )


def metrics(c):
    TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
    n = TP + FP + FN + TN
    out = {"n": n, **{k: c[k] for k in ("TP", "TN", "FP", "FN", "unsure")}}

    prec_lo, prec_hi = wilson(TP, TP + FP)
    npv_lo, npv_hi = wilson(TN, TN + FN)
    out["precision"] = TP / (TP + FP) if TP + FP else float("nan")
    out["precision_ci_low"], out["precision_ci_high"] = prec_lo, prec_hi
    out["npv"] = TN / (TN + FN) if TN + FN else float("nan")
    out["npv_ci_low"], out["npv_ci_high"] = npv_lo, npv_hi

    # Sample-only. Retained for transparency, not for reporting: see docstring.
    po = (TP + TN) / n
    pe = ((TP + FP) / n) * ((TP + FN) / n) + ((TN + FN) / n) * ((TN + FP) / n)
    out["raw_agreement_SAMPLE_ONLY"] = po
    out["cohens_kappa_SAMPLE_ONLY"] = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return out


def report(label, m):
    print(f"\n--- {label} (n = {m['n']}) ---")
    print(f"  admitted by the rule : {m['TP'] + m['FP']:>3}"
          f"   ({m['TP']} confirmed, {m['FP']} false positives)")
    print(f"  rejected by the rule : {m['TN'] + m['FN']:>3}"
          f"   ({m['TN']} correctly excluded, {m['FN']} false negatives)")
    if m["unsure"]:
        print(f"  marked unsure        : {m['unsure']} (excluded)")
    print(f"  precision            : {m['precision']:.4f}"
          f"  [{m['precision_ci_low']:.3f}, {m['precision_ci_high']:.3f}]   generalises")
    print(f"  NPV                  : {m['npv']:.4f}"
          f"  [{m['npv_ci_low']:.3f}, {m['npv_ci_high']:.3f}]   generalises")
    print(f"  raw agreement        : {m['raw_agreement_SAMPLE_ONLY']:.4f}"
          "                    sample only")
    print(f"  Cohen's kappa        : {m['cohens_kappa_SAMPLE_ONLY']:.4f}"
          "                    sample only")


def main():
    df = load_sheet(SHEET)
    dupes = df[df.project_id.duplicated(keep=False)]

    rows = []
    m_all = metrics(confusion(df))
    report("as coded", m_all)
    rows.append({"basis": "as_coded", **m_all})

    if len(dupes):
        print(f"\n  NOTE: {df.project_id.duplicated().sum()} duplicated project_id "
              f"({df.project_id.nunique()} distinct projects in {len(df)} rows)")
        m_dedup = metrics(confusion(df.drop_duplicates(subset="project_id")))
        report("deduplicated", m_dedup)
        rows.append({"basis": "deduplicated", **m_dedup})

    VAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(VAL / "screening_validation_summary.csv", index=False)

    c = confusion(df)
    pd.DataFrame(
        [[c["TP"], c["FP"]], [c["FN"], c["TN"]]],
        index=["rule: keep", "rule: drop"],
        columns=["human: keep", "human: drop"],
    ).to_csv(VAL / "screening_validation_confusion.csv")

    print(f"\nWrote screening_validation_summary.csv and "
          f"screening_validation_confusion.csv to {VAL.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
