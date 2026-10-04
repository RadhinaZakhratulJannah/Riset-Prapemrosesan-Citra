#!/usr/bin/env python3
"""
Uji statistik paired ABMIL vs CLAM:
1) DeLong test untuk membandingkan AUROC.
2) McNemar test untuk membandingkan hasil klasifikasi berpasangan.

DEFAULT:
- Menggunakan OOF prediction dari 5-fold CV.
- TIDAK mengevaluasi TEST.
- McNemar dihitung pada:
  a) threshold 0.5 untuk kedua model
  b) threshold mutant-priority masing-masing model yang sudah dikunci dari CV

Input default:
  /workspace/TCGA-LUAD/ABMIL_MutantPriority/oof_predictions.csv
  /workspace/TCGA-LUAD/CLAM_SB_MutantPriority_Final/oof_predictions.csv
  /workspace/TCGA-LUAD/ABMIL_MutantPriority/cv_aggregate.json
  /workspace/TCGA-LUAD/CLAM_SB_MutantPriority_Final/cv_aggregate.json

Output default:
  /workspace/TCGA-LUAD/ABMIL_vs_CLAM_StatTests/

Catatan metodologi:
- DeLong menggunakan probabilitas kontinu dari sampel yang sama.
- McNemar menggunakan prediksi biner dari sampel yang sama.
- Hasil OOF adalah analisis development/CV. Untuk inferensi final, ulangi uji
  pada independent TEST setelah hyperparameter dan threshold benar-benar dikunci.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, binomtest
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Utility
# -----------------------------

def read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")
    with open(path, "r") as f:
        return json.load(f)


def find_prob_col(df: pd.DataFrame, model_name: str):
    candidates = [
        "prob_mutant",
        "probability",
        "prob",
        "y_prob",
        "score",
        "pred_prob",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Kolom probabilitas {model_name} tidak ditemukan. "
        f"Kolom yang tersedia: {list(df.columns)}"
    )


def validate_binary_target(y):
    vals = sorted(pd.Series(y).dropna().unique().tolist())
    if vals != [0, 1]:
        raise SystemExit(f"Target harus biner 0/1. Ditemukan: {vals}")


# -----------------------------
# DeLong test
# -----------------------------

def compute_midrank(x):
    """Compute midranks, 1-based as required by fast DeLong."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]

    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1

        # Midrank for positions i..j-1 in 1-based indexing.
        mid = 0.5 * (i + j - 1) + 1.0
        midranks[i:j] = mid
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong covariance for correlated ROC AUCs.

    predictions_sorted_transposed:
      shape (n_models, n_samples), with positive samples first.
    """
    preds = np.asarray(predictions_sorted_transposed, dtype=float)
    k = preds.shape[0]
    m = int(label_1_count)
    n = preds.shape[1] - m

    if m < 2 or n < 2:
        raise ValueError("DeLong memerlukan minimal 2 positif dan 2 negatif.")

    positive = preds[:, :m]
    negative = preds[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(preds[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    # Ensure 2D covariance even in edge cases.
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)

    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, pred_a, pred_b, alpha=0.05):
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)

    validate_binary_target(y)

    # Positive samples first, then negative.
    order = np.argsort(-y)
    y_sorted = y[order]
    m = int(y_sorted.sum())

    preds = np.vstack([pa[order], pb[order]])
    aucs, cov = fast_delong(preds, m)

    contrast = np.array([1.0, -1.0])
    var_diff = float(contrast @ cov @ contrast.T)
    diff = float(aucs[0] - aucs[1])

    if var_diff <= 0 or not np.isfinite(var_diff):
        se = 0.0
        z = np.inf if diff != 0 else 0.0
        p = 0.0 if diff != 0 else 1.0
        ci_low = diff
        ci_high = diff
    else:
        se = math.sqrt(var_diff)
        z = abs(diff) / se
        p = float(2.0 * norm.sf(z))
        zcrit = float(norm.ppf(1.0 - alpha / 2.0))
        ci_low = diff - zcrit * se
        ci_high = diff + zcrit * se

    # sklearn AUC used as cross-check.
    auc_a_check = float(roc_auc_score(y, pa))
    auc_b_check = float(roc_auc_score(y, pb))

    return {
        "n": int(len(y)),
        "n_mutant": int((y == 1).sum()),
        "n_wildtype": int((y == 0).sum()),
        "abmil_auc": float(aucs[0]),
        "clam_auc": float(aucs[1]),
        "abmil_auc_sklearn_check": auc_a_check,
        "clam_auc_sklearn_check": auc_b_check,
        "auc_difference_abmil_minus_clam": diff,
        "se_difference": float(se),
        "z": float(z),
        "p_value_two_sided": float(p),
        "difference_ci95_low": float(ci_low),
        "difference_ci95_high": float(ci_high),
        "alpha": float(alpha),
        "significant_at_alpha": bool(p < alpha),
    }


# -----------------------------
# McNemar test
# -----------------------------

def mcnemar_test_from_predictions(y_true, pred_a, pred_b, alpha=0.05):
    """
    Compares paired correctness:
      b = ABMIL correct, CLAM wrong
      c = ABMIL wrong, CLAM correct
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    bpred = np.asarray(pred_b, dtype=int)

    correct_a = (a == y)
    correct_b = (bpred == y)

    both_correct = int(np.sum(correct_a & correct_b))
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        exact_p = 1.0
        chi2_cc = 0.0
        chi2_cc_p = 1.0
        chi2_uncorrected = 0.0
        chi2_uncorrected_p = 1.0
    else:
        exact_p = float(
            binomtest(
                min(a_correct_b_wrong, a_wrong_b_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

        chi2_cc = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1.0) ** 2 / discordant
        chi2_cc_p = float(chi2.sf(chi2_cc, df=1))

        chi2_uncorrected = (
            (a_correct_b_wrong - a_wrong_b_correct) ** 2 / discordant
        )
        chi2_uncorrected_p = float(chi2.sf(chi2_uncorrected, df=1))

    # Use exact McNemar as primary because it is valid for small discordant counts.
    primary_p = exact_p

    return {
        "n": int(len(y)),
        "both_correct": both_correct,
        "abmil_correct_clam_wrong": a_correct_b_wrong,
        "abmil_wrong_clam_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": int(discordant),
        "mcnemar_exact_p": exact_p,
        "mcnemar_chi2_continuity_corrected": float(chi2_cc),
        "mcnemar_chi2_cc_p": chi2_cc_p,
        "mcnemar_chi2_uncorrected": float(chi2_uncorrected),
        "mcnemar_chi2_uncorrected_p": chi2_uncorrected_p,
        "primary_p_value": primary_p,
        "alpha": float(alpha),
        "significant_at_alpha": bool(primary_p < alpha),
        "abmil_accuracy": float(correct_a.mean()),
        "clam_accuracy": float(correct_b.mean()),
    }


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = (tp + tn) / len(y) if len(y) else np.nan

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
    }


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/workspace/TCGA-LUAD"),
    )
    ap.add_argument(
        "--abmil-dir",
        default="ABMIL_MutantPriority",
    )
    ap.add_argument(
        "--clam-dir",
        default="CLAM_SB_MutantPriority_Final",
    )
    ap.add_argument(
        "--out-dir",
        default="ABMIL_vs_CLAM_StatTests",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )
    ap.add_argument(
        "--abmil-pred",
        type=Path,
        default=None,
        help="Opsional: CSV prediction ABMIL. Default = oof_predictions.csv",
    )
    ap.add_argument(
        "--clam-pred",
        type=Path,
        default=None,
        help="Opsional: CSV prediction CLAM. Default = oof_predictions.csv",
    )

    args = ap.parse_args()

    base = args.base_dir
    ab_dir = base / args.abmil_dir
    cl_dir = base / args.clam_dir
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ab_pred_path = args.abmil_pred or (ab_dir / "oof_predictions.csv")
    cl_pred_path = args.clam_pred or (cl_dir / "oof_predictions.csv")

    if not ab_pred_path.exists():
        raise SystemExit(f"Prediction ABMIL tidak ditemukan: {ab_pred_path}")
    if not cl_pred_path.exists():
        raise SystemExit(f"Prediction CLAM tidak ditemukan: {cl_pred_path}")

    ab = pd.read_csv(ab_pred_path)
    cl = pd.read_csv(cl_pred_path)

    for name, df in (("ABMIL", ab), ("CLAM", cl)):
        if "slide" not in df.columns or "target" not in df.columns:
            raise SystemExit(
                f"{name} harus memiliki kolom slide dan target. "
                f"Kolom tersedia: {list(df.columns)}"
            )

    ab_prob_col = find_prob_col(ab, "ABMIL")
    cl_prob_col = find_prob_col(cl, "CLAM")

    a_cols = ["slide", "target", ab_prob_col]
    c_cols = ["slide", "target", cl_prob_col]

    if "fold" in ab.columns:
        a_cols.append("fold")
    if "fold" in cl.columns:
        c_cols.append("fold")

    a = ab[a_cols].copy().rename(
        columns={
            "target": "target_abmil",
            ab_prob_col: "prob_abmil",
            "fold": "fold_abmil",
        }
    )
    c = cl[c_cols].copy().rename(
        columns={
            "target": "target_clam",
            cl_prob_col: "prob_clam",
            "fold": "fold_clam",
        }
    )

    paired = a.merge(c, on="slide", how="outer", indicator=True)

    if not (paired["_merge"] == "both").all():
        only_a = paired.loc[paired["_merge"] == "left_only", "slide"].tolist()
        only_c = paired.loc[paired["_merge"] == "right_only", "slide"].tolist()
        raise SystemExit(
            "Slide ABMIL dan CLAM tidak sama.\n"
            f"Hanya ABMIL: {only_a[:10]}\n"
            f"Hanya CLAM: {only_c[:10]}"
        )

    if not (paired["target_abmil"] == paired["target_clam"]).all():
        raise SystemExit("Target ABMIL dan CLAM tidak sama untuk sebagian slide.")

    if "fold_abmil" in paired.columns and "fold_clam" in paired.columns:
        if not (paired["fold_abmil"] == paired["fold_clam"]).all():
            raise SystemExit("Fold ABMIL dan CLAM tidak identik.")

    paired = paired.sort_values("slide").reset_index(drop=True)
    y = paired["target_abmil"].astype(int).to_numpy()
    pa = paired["prob_abmil"].astype(float).to_numpy()
    pc = paired["prob_clam"].astype(float).to_numpy()

    validate_binary_target(y)

    # Threshold mutant-priority learned during CV.
    ab_agg = read_json(ab_dir / "cv_aggregate.json")
    cl_agg = read_json(cl_dir / "cv_aggregate.json")

    thr_ab = float(ab_agg["selected_oof_threshold_development"])
    thr_cl = float(cl_agg["selected_oof_threshold_development"])

    # -------------------------
    # 1. DeLong
    # -------------------------
    delong = delong_roc_test(y, pa, pc, alpha=args.alpha)

    # Add AUPRC descriptively (not DeLong-tested).
    delong["abmil_auprc_descriptive"] = float(average_precision_score(y, pa))
    delong["clam_auprc_descriptive"] = float(average_precision_score(y, pc))

    with open(out_dir / "delong_test.json", "w") as f:
        json.dump(delong, f, indent=2)

    # -------------------------
    # 2. McNemar @ 0.5
    # -------------------------
    pred_a_05 = (pa >= 0.5).astype(int)
    pred_c_05 = (pc >= 0.5).astype(int)

    mc_05 = mcnemar_test_from_predictions(
        y, pred_a_05, pred_c_05, alpha=args.alpha
    )
    mc_05["comparison"] = "threshold_0.5_both"
    mc_05["abmil_threshold"] = 0.5
    mc_05["clam_threshold"] = 0.5

    # -------------------------
    # 3. McNemar @ mutant-priority thresholds
    # -------------------------
    pred_a_mp = (pa >= thr_ab).astype(int)
    pred_c_mp = (pc >= thr_cl).astype(int)

    mc_mp = mcnemar_test_from_predictions(
        y, pred_a_mp, pred_c_mp, alpha=args.alpha
    )
    mc_mp["comparison"] = "mutant_priority_thresholds_development"
    mc_mp["abmil_threshold"] = thr_ab
    mc_mp["clam_threshold"] = thr_cl

    # -------------------------
    # 4. Exploratory class-specific McNemar
    # -------------------------
    mutant_mask = (y == 1)
    wt_mask = (y == 0)

    mc_mutant = mcnemar_test_from_predictions(
        y[mutant_mask],
        pred_a_mp[mutant_mask],
        pred_c_mp[mutant_mask],
        alpha=args.alpha,
    )
    mc_mutant["comparison"] = "mutant_only_at_mutant_priority_thresholds"
    mc_mutant["abmil_threshold"] = thr_ab
    mc_mutant["clam_threshold"] = thr_cl
    mc_mutant["exploratory"] = True

    mc_wt = mcnemar_test_from_predictions(
        y[wt_mask],
        pred_a_mp[wt_mask],
        pred_c_mp[wt_mask],
        alpha=args.alpha,
    )
    mc_wt["comparison"] = "wildtype_only_at_mutant_priority_thresholds"
    mc_wt["abmil_threshold"] = thr_ab
    mc_wt["clam_threshold"] = thr_cl
    mc_wt["exploratory"] = True

    mcnemar_rows = [mc_05, mc_mp, mc_mutant, mc_wt]
    pd.DataFrame(mcnemar_rows).to_csv(
        out_dir / "mcnemar_tests.csv", index=False
    )

    # Add binary predictions + correctness to paired output.
    paired["target"] = y

    paired["pred_abmil_0.5"] = pred_a_05
    paired["pred_clam_0.5"] = pred_c_05
    paired["abmil_correct_0.5"] = (pred_a_05 == y).astype(int)
    paired["clam_correct_0.5"] = (pred_c_05 == y).astype(int)

    paired["pred_abmil_mutant_priority"] = pred_a_mp
    paired["pred_clam_mutant_priority"] = pred_c_mp
    paired["abmil_correct_mutant_priority"] = (pred_a_mp == y).astype(int)
    paired["clam_correct_mutant_priority"] = (pred_c_mp == y).astype(int)

    paired.to_csv(
        out_dir / "paired_predictions_with_correctness.csv",
        index=False
    )

    # Descriptive binary metrics.
    bm_a_05 = binary_metrics(y, pred_a_05)
    bm_c_05 = binary_metrics(y, pred_c_05)
    bm_a_mp = binary_metrics(y, pred_a_mp)
    bm_c_mp = binary_metrics(y, pred_c_mp)

    summary = {
        "analysis_set": "OOF CV development predictions"
            if args.abmil_pred is None and args.clam_pred is None
            else "user-specified paired prediction files",
        "n": int(len(y)),
        "n_mutant": int((y == 1).sum()),
        "n_wildtype": int((y == 0).sum()),
        "alpha": float(args.alpha),
        "delong": delong,
        "mcnemar_threshold_0.5": mc_05,
        "mcnemar_mutant_priority_thresholds": mc_mp,
        "exploratory_mcnemar_mutant_only": mc_mutant,
        "exploratory_mcnemar_wildtype_only": mc_wt,
        "binary_metrics_0.5": {
            "ABMIL": bm_a_05,
            "CLAM": bm_c_05,
        },
        "binary_metrics_mutant_priority": {
            "ABMIL": bm_a_mp,
            "CLAM": bm_c_mp,
        },
        "thresholds": {
            "ABMIL_mutant_priority": thr_ab,
            "CLAM_mutant_priority": thr_cl,
        },
    }

    with open(out_dir / "statistical_tests_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------
    # Markdown report
    # -------------------------
    def sig_text(p):
        return "signifikan" if p < args.alpha else "tidak signifikan"

    md = []
    md.append("# Uji Statistik ABMIL vs CLAM")
    md.append("")
    md.append("Analisis default menggunakan **OOF 5-fold CV (development)**, bukan TEST.")
    md.append("")

    md.append("## 1. DeLong test — AUROC")
    md.append("")
    md.append("| Item | Hasil |")
    md.append("|---|---:|")
    md.append(f"| N | {delong['n']} |")
    md.append(f"| Mutant | {delong['n_mutant']} |")
    md.append(f"| Wild-Type | {delong['n_wildtype']} |")
    md.append(f"| AUROC ABMIL | {delong['abmil_auc']:.4f} |")
    md.append(f"| AUROC CLAM | {delong['clam_auc']:.4f} |")
    md.append(
        f"| Selisih AUROC (ABMIL−CLAM) | "
        f"{delong['auc_difference_abmil_minus_clam']:+.4f} |"
    )
    md.append(
        f"| 95% CI selisih | "
        f"{delong['difference_ci95_low']:.4f} s.d. "
        f"{delong['difference_ci95_high']:.4f} |"
    )
    md.append(f"| z | {delong['z']:.4f} |")
    md.append(f"| p-value DeLong | {delong['p_value_two_sided']:.6f} |")
    md.append(
        f"| Keputusan α={args.alpha:.2f} | "
        f"{sig_text(delong['p_value_two_sided'])} |"
    )
    md.append("")

    md.append("## 2. McNemar — threshold 0.5")
    md.append("")
    md.append("| Item | Hasil |")
    md.append("|---|---:|")
    md.append(f"| ABMIL benar, CLAM salah | {mc_05['abmil_correct_clam_wrong']} |")
    md.append(f"| ABMIL salah, CLAM benar | {mc_05['abmil_wrong_clam_correct']} |")
    md.append(f"| Discordant pairs | {mc_05['discordant_pairs']} |")
    md.append(f"| Exact McNemar p | {mc_05['mcnemar_exact_p']:.6f} |")
    md.append(
        f"| Continuity-corrected p | {mc_05['mcnemar_chi2_cc_p']:.6f} |"
    )
    md.append(
        f"| Keputusan α={args.alpha:.2f} | "
        f"{sig_text(mc_05['primary_p_value'])} |"
    )
    md.append("")

    md.append("## 3. McNemar — threshold Mutant-Priority")
    md.append("")
    md.append(f"- Threshold ABMIL: **{thr_ab:.6f}**")
    md.append(f"- Threshold CLAM: **{thr_cl:.6f}**")
    md.append("")
    md.append("| Item | Hasil |")
    md.append("|---|---:|")
    md.append(f"| ABMIL benar, CLAM salah | {mc_mp['abmil_correct_clam_wrong']} |")
    md.append(f"| ABMIL salah, CLAM benar | {mc_mp['abmil_wrong_clam_correct']} |")
    md.append(f"| Discordant pairs | {mc_mp['discordant_pairs']} |")
    md.append(f"| Exact McNemar p | {mc_mp['mcnemar_exact_p']:.6f} |")
    md.append(
        f"| Continuity-corrected p | {mc_mp['mcnemar_chi2_cc_p']:.6f} |"
    )
    md.append(
        f"| Keputusan α={args.alpha:.2f} | "
        f"{sig_text(mc_mp['primary_p_value'])} |"
    )
    md.append("")

    md.append("## 4. Exploratory — per kelas pada threshold Mutant-Priority")
    md.append("")
    md.append("| Populasi | N | Exact McNemar p | Keterangan |")
    md.append("|---|---:|---:|---|")
    md.append(
        f"| Mutant saja | {mc_mutant['n']} | "
        f"{mc_mutant['mcnemar_exact_p']:.6f} | Perbedaan sensitivitas secara paired |"
    )
    md.append(
        f"| Wild-Type saja | {mc_wt['n']} | "
        f"{mc_wt['mcnemar_exact_p']:.6f} | Perbedaan spesifisitas secara paired |"
    )
    md.append("")

    md.append("## Interpretasi")
    md.append("")
    md.append(
        "- DeLong menguji apakah perbedaan **AUROC** ABMIL dan CLAM "
        "lebih besar daripada yang diharapkan akibat variasi sampel."
    )
    md.append(
        "- McNemar menguji apakah kedua model memiliki perbedaan pada "
        "**hasil klasifikasi benar/salah untuk slide yang sama**."
    )
    md.append(
        "- p < 0.05 berarti terdapat bukti perbedaan statistik pada pengujian tersebut; "
        "p ≥ 0.05 berarti data ini belum memberikan bukti yang cukup untuk menyatakan "
        "perbedaan statistik."
    )
    md.append(
        "- Karena analisis default menggunakan OOF CV dan threshold juga dikembangkan "
        "dari OOF, hasil ini sebaiknya diperlakukan sebagai analisis development. "
        "Uji final sebaiknya diulang pada independent TEST setelah semua pilihan model "
        "dan threshold dikunci."
    )
    md.append("")

    (out_dir / "statistical_tests_summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    # -------------------------
    # Terminal output
    # -------------------------
    print("=" * 100)
    print("UJI STATISTIK ABMIL vs CLAM - OOF 5-FOLD CV")
    print("=" * 100)

    print("\n[1] DeLong Test - AUROC")
    print(f"N                 : {len(y)} ({(y==1).sum()} Mutant, {(y==0).sum()} Wild-Type)")
    print(f"ABMIL AUROC       : {delong['abmil_auc']:.4f}")
    print(f"CLAM AUROC        : {delong['clam_auc']:.4f}")
    print(
        f"Delta ABMIL-CLAM  : "
        f"{delong['auc_difference_abmil_minus_clam']:+.4f}"
    )
    print(
        f"95% CI Delta      : "
        f"{delong['difference_ci95_low']:.4f} to "
        f"{delong['difference_ci95_high']:.4f}"
    )
    print(f"z                 : {delong['z']:.4f}")
    print(f"p-value           : {delong['p_value_two_sided']:.6f}")
    print(
        f"Keputusan         : {sig_text(delong['p_value_two_sided'])} "
        f"(alpha={args.alpha})"
    )

    print("\n[2] McNemar - threshold 0.5")
    print(f"ABMIL benar / CLAM salah : {mc_05['abmil_correct_clam_wrong']}")
    print(f"ABMIL salah / CLAM benar : {mc_05['abmil_wrong_clam_correct']}")
    print(f"Discordant pairs         : {mc_05['discordant_pairs']}")
    print(f"Exact p                  : {mc_05['mcnemar_exact_p']:.6f}")
    print(f"Corrected chi-square p   : {mc_05['mcnemar_chi2_cc_p']:.6f}")
    print(
        f"Keputusan                : "
        f"{sig_text(mc_05['primary_p_value'])}"
    )

    print("\n[3] McNemar - mutant-priority thresholds")
    print(f"ABMIL threshold          : {thr_ab:.6f}")
    print(f"CLAM threshold           : {thr_cl:.6f}")
    print(f"ABMIL benar / CLAM salah : {mc_mp['abmil_correct_clam_wrong']}")
    print(f"ABMIL salah / CLAM benar : {mc_mp['abmil_wrong_clam_correct']}")
    print(f"Discordant pairs         : {mc_mp['discordant_pairs']}")
    print(f"Exact p                  : {mc_mp['mcnemar_exact_p']:.6f}")
    print(f"Corrected chi-square p   : {mc_mp['mcnemar_chi2_cc_p']:.6f}")
    print(
        f"Keputusan                : "
        f"{sig_text(mc_mp['primary_p_value'])}"
    )

    print("\n[4] Exploratory per-class @ mutant-priority")
    print(
        f"Mutant-only McNemar exact p   : "
        f"{mc_mutant['mcnemar_exact_p']:.6f}"
    )
    print(
        f"Wild-Type-only McNemar exact p: "
        f"{mc_wt['mcnemar_exact_p']:.6f}"
    )

    print("\nOutput:")
    print(f"  {out_dir / 'delong_test.json'}")
    print(f"  {out_dir / 'mcnemar_tests.csv'}")
    print(f"  {out_dir / 'paired_predictions_with_correctness.csv'}")
    print(f"  {out_dir / 'statistical_tests_summary.json'}")
    print(f"  {out_dir / 'statistical_tests_summary.md'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
