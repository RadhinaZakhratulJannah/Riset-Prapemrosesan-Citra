#!/usr/bin/env python3
"""
FINAL TEST statistical comparison: ABMIL vs CLAM

Primary analyses:
1) Paired DeLong test on ensemble probabilities -> compares AUROC.
2) Exact McNemar test on LOCKED-threshold binary predictions.

IMPORTANT:
- Reads the already-opened independent 40-slide TEST predictions.
- Does NOT train, tune, change thresholds, or select checkpoints.
- Uses the locked predictions already written by final_test_evaluation_abmil_clam.py.

Default input:
  /workspace/TCGA-LUAD/Final_Test_ABMIL_vs_CLAM/final_test_paired_predictions.csv

Default output:
  /workspace/TCGA-LUAD/Final_Test_ABMIL_vs_CLAM/Statistical_Tests_Final/
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, binomtest
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------

def compute_midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    mr = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        mr[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    out = np.empty(n, dtype=float)
    out[order] = mr
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    preds = np.asarray(predictions_sorted_transposed, dtype=float)
    k = preds.shape[0]
    m = int(label_1_count)
    n = preds.shape[1] - m

    if m < 2 or n < 2:
        raise ValueError("DeLong memerlukan minimal 2 Mutant dan 2 Wild-Type.")

    pos = preds[:, :m]
    neg = preds[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(preds[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n

    return aucs, cov


def delong_test(y, prob_abmil, prob_clam, alpha=0.05):
    y = np.asarray(y, dtype=int)
    pa = np.asarray(prob_abmil, dtype=float)
    pc = np.asarray(prob_clam, dtype=float)

    values = sorted(np.unique(y).tolist())
    if values != [0, 1]:
        raise ValueError(f"Target harus 0/1, ditemukan: {values}")

    order = np.argsort(-y)  # positive first
    m = int(y.sum())

    preds = np.vstack([pa[order], pc[order]])
    aucs, cov = fast_delong(preds, m)

    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast.T)
    diff = float(aucs[0] - aucs[1])

    if variance <= 0 or not np.isfinite(variance):
        se = 0.0
        z = 0.0 if diff == 0 else float("inf")
        p = 1.0 if diff == 0 else 0.0
        ci_low = diff
        ci_high = diff
    else:
        se = math.sqrt(variance)
        z = diff / se
        p = float(2 * norm.sf(abs(z)))
        zcrit = float(norm.ppf(1 - alpha / 2))
        ci_low = diff - zcrit * se
        ci_high = diff + zcrit * se

    return {
        "n": int(len(y)),
        "n_mutant": int((y == 1).sum()),
        "n_wildtype": int((y == 0).sum()),
        "abmil_auroc": float(aucs[0]),
        "clam_auroc": float(aucs[1]),
        "abmil_auroc_sklearn_check": float(roc_auc_score(y, pa)),
        "clam_auroc_sklearn_check": float(roc_auc_score(y, pc)),
        "delta_auroc_abmil_minus_clam": diff,
        "delta_se": float(se),
        "delta_ci95_low": float(ci_low),
        "delta_ci95_high": float(ci_high),
        "z": float(z),
        "p_value_two_sided": float(p),
        "alpha": float(alpha),
        "significant": bool(p < alpha),
    }


# ---------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------

def mcnemar_exact(y, pred_abmil, pred_clam, alpha=0.05):
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_abmil, dtype=int)
    c = np.asarray(pred_clam, dtype=int)

    a_ok = (a == y)
    c_ok = (c == y)

    both_correct = int(np.sum(a_ok & c_ok))
    ab_correct_cl_wrong = int(np.sum(a_ok & ~c_ok))
    ab_wrong_cl_correct = int(np.sum(~a_ok & c_ok))
    both_wrong = int(np.sum(~a_ok & ~c_ok))

    discordant = ab_correct_cl_wrong + ab_wrong_cl_correct

    if discordant == 0:
        p_exact = 1.0
        chi2_cc = 0.0
        p_cc = 1.0
    else:
        # Two-sided exact McNemar = exact binomial test on discordant pairs.
        p_exact = float(
            binomtest(
                ab_correct_cl_wrong,
                n=discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )

        chi2_cc = (
            (abs(ab_correct_cl_wrong - ab_wrong_cl_correct) - 1.0) ** 2
            / discordant
        )
        p_cc = float(chi2.sf(chi2_cc, df=1))

    return {
        "n": int(len(y)),
        "both_correct": both_correct,
        "abmil_correct_clam_wrong": ab_correct_cl_wrong,
        "abmil_wrong_clam_correct": ab_wrong_cl_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": int(discordant),
        "exact_p_value": p_exact,
        "continuity_corrected_chi2": float(chi2_cc),
        "continuity_corrected_p_value": p_cc,
        "alpha": float(alpha),
        "significant": bool(p_exact < alpha),
        "abmil_accuracy": float(a_ok.mean()),
        "clam_accuracy": float(c_ok.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/workspace/TCGA-LUAD/Final_Test_ABMIL_vs_CLAM/"
            "final_test_paired_predictions.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/TCGA-LUAD/Final_Test_ABMIL_vs_CLAM/"
            "Statistical_Tests_Final"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input tidak ditemukan: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = [
        "slide",
        "target",
        "abmil_prob_mutant",
        "clam_prob_mutant",
        "abmil_threshold_locked",
        "clam_threshold_locked",
        "abmil_pred_mutant",
        "clam_pred_mutant",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Kolom berikut tidak ditemukan: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    if df["slide"].duplicated().any():
        raise SystemExit("Ada slide duplikat pada FINAL TEST.")

    y = df["target"].astype(int).to_numpy()
    pa = df["abmil_prob_mutant"].astype(float).to_numpy()
    pc = df["clam_prob_mutant"].astype(float).to_numpy()
    pred_a = df["abmil_pred_mutant"].astype(int).to_numpy()
    pred_c = df["clam_pred_mutant"].astype(int).to_numpy()

    # Verify that locked predictions are internally consistent.
    thr_a_values = df["abmil_threshold_locked"].astype(float).unique()
    thr_c_values = df["clam_threshold_locked"].astype(float).unique()

    if len(thr_a_values) != 1 or len(thr_c_values) != 1:
        raise SystemExit("Threshold locked tidak konsisten di seluruh baris.")

    thr_a = float(thr_a_values[0])
    thr_c = float(thr_c_values[0])

    expected_a = (pa >= thr_a).astype(int)
    expected_c = (pc >= thr_c).astype(int)

    if not np.array_equal(expected_a, pred_a):
        raise SystemExit("Prediksi ABMIL tidak sesuai threshold locked.")
    if not np.array_equal(expected_c, pred_c):
        raise SystemExit("Prediksi CLAM tidak sesuai threshold locked.")

    if len(df) != 40:
        print(f"WARNING: jumlah TEST = {len(df)}, bukan 40.")

    # Primary final tests.
    dl = delong_test(y, pa, pc, alpha=args.alpha)
    mc = mcnemar_exact(y, pred_a, pred_c, alpha=args.alpha)

    # Exploratory class-specific McNemar.
    mut = (y == 1)
    wt = (y == 0)

    mc_mut = mcnemar_exact(
        y[mut], pred_a[mut], pred_c[mut], alpha=args.alpha
    )
    mc_wt = mcnemar_exact(
        y[wt], pred_a[wt], pred_c[wt], alpha=args.alpha
    )

    payload = {
        "analysis": "INDEPENDENT FINAL TEST - LOCKED MODELS AND THRESHOLDS",
        "input": str(args.input),
        "thresholds_locked": {
            "ABMIL": thr_a,
            "CLAM": thr_c,
        },
        "delong_primary": dl,
        "mcnemar_primary_locked_thresholds": mc,
        "mcnemar_exploratory_mutant_only": mc_mut,
        "mcnemar_exploratory_wildtype_only": mc_wt,
    }

    with open(args.out_dir / "final_statistical_tests.json", "w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame([
        {
            "analysis": "All TEST - locked thresholds",
            **mc,
        },
        {
            "analysis": "Mutant only - exploratory",
            **mc_mut,
        },
        {
            "analysis": "Wild-Type only - exploratory",
            **mc_wt,
        },
    ]).to_csv(args.out_dir / "final_mcnemar_results.csv", index=False)

    decision_d = (
        "SIGNIFIKAN" if dl["significant"] else "TIDAK SIGNIFIKAN"
    )
    decision_m = (
        "SIGNIFIKAN" if mc["significant"] else "TIDAK SIGNIFIKAN"
    )

    md = f"""# Final Statistical Tests — ABMIL vs CLAM

**Dataset:** independent FINAL TEST  
**N:** {len(y)} ({int(mut.sum())} Mutant, {int(wt.sum())} Wild-Type)  
**Threshold ABMIL locked:** {thr_a:.6f}  
**Threshold CLAM locked:** {thr_c:.6f}  

## 1. Paired DeLong test — AUROC

| Item | Result |
|---|---:|
| ABMIL AUROC | {dl['abmil_auroc']:.4f} |
| CLAM AUROC | {dl['clam_auroc']:.4f} |
| Δ AUROC (ABMIL−CLAM) | {dl['delta_auroc_abmil_minus_clam']:+.4f} |
| 95% CI Δ AUROC | {dl['delta_ci95_low']:.4f} to {dl['delta_ci95_high']:.4f} |
| z | {dl['z']:.4f} |
| p-value | {dl['p_value_two_sided']:.6f} |
| Decision α={args.alpha:.2f} | **{decision_d}** |

## 2. Exact McNemar test — locked thresholds

| Item | Result |
|---|---:|
| Both correct | {mc['both_correct']} |
| ABMIL correct, CLAM wrong | {mc['abmil_correct_clam_wrong']} |
| ABMIL wrong, CLAM correct | {mc['abmil_wrong_clam_correct']} |
| Both wrong | {mc['both_wrong']} |
| Discordant pairs | {mc['discordant_pairs']} |
| Exact p-value | {mc['exact_p_value']:.6f} |
| Continuity-corrected p-value | {mc['continuity_corrected_p_value']:.6f} |
| Decision α={args.alpha:.2f} | **{decision_m}** |

## 3. Exploratory class-specific McNemar

| Class | N | Exact p-value |
|---|---:|---:|
| Mutant | {mc_mut['n']} | {mc_mut['exact_p_value']:.6f} |
| Wild-Type | {mc_wt['n']} | {mc_wt['exact_p_value']:.6f} |

## Interpretation rule

- p < 0.05: evidence of a statistically significant difference for that test.
- p ≥ 0.05: insufficient evidence of a statistically significant difference.
- DeLong evaluates AUROC differences using continuous probabilities.
- McNemar evaluates paired correct/incorrect classifications at the already locked thresholds.
- The class-specific tests are exploratory because the class sample sizes are small.
"""

    (args.out_dir / "final_statistical_tests_summary.md").write_text(
        md, encoding="utf-8"
    )

    print("=" * 100)
    print("FINAL TEST STATISTICS - ABMIL vs CLAM")
    print("=" * 100)
    print(
        f"N = {len(y)} | Mutant = {int(mut.sum())} | "
        f"Wild-Type = {int(wt.sum())}"
    )
    print(f"ABMIL locked threshold = {thr_a:.6f}")
    print(f"CLAM  locked threshold = {thr_c:.6f}")

    print("\n[1] Paired DeLong - FINAL TEST AUROC")
    print(f"ABMIL AUROC          : {dl['abmil_auroc']:.4f}")
    print(f"CLAM AUROC           : {dl['clam_auroc']:.4f}")
    print(
        f"Delta ABMIL-CLAM     : "
        f"{dl['delta_auroc_abmil_minus_clam']:+.4f}"
    )
    print(
        f"95% CI Delta         : "
        f"{dl['delta_ci95_low']:.4f} to {dl['delta_ci95_high']:.4f}"
    )
    print(f"p-value              : {dl['p_value_two_sided']:.6f}")
    print(f"Decision             : {decision_d}")

    print("\n[2] Exact McNemar - FINAL locked thresholds")
    print(
        f"ABMIL correct / CLAM wrong : "
        f"{mc['abmil_correct_clam_wrong']}"
    )
    print(
        f"ABMIL wrong / CLAM correct : "
        f"{mc['abmil_wrong_clam_correct']}"
    )
    print(f"Discordant pairs            : {mc['discordant_pairs']}")
    print(f"Exact p-value               : {mc['exact_p_value']:.6f}")
    print(f"Decision                    : {decision_m}")

    print("\n[3] Exploratory per class")
    print(
        f"Mutant exact p              : "
        f"{mc_mut['exact_p_value']:.6f}"
    )
    print(
        f"Wild-Type exact p           : "
        f"{mc_wt['exact_p_value']:.6f}"
    )

    print("\nOutput:")
    print(args.out_dir / "final_statistical_tests.json")
    print(args.out_dir / "final_mcnemar_results.csv")
    print(args.out_dir / "final_statistical_tests_summary.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
