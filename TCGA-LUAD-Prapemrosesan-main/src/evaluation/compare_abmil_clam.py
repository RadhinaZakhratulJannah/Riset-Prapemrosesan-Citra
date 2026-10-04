#!/usr/bin/env python3
"""
Buat tabel perbandingan hasil CV ABMIL vs CLAM Mutant-Priority.

Tidak melatih model dan tidak mengevaluasi TEST.
Input default:
  /workspace/TCGA-LUAD/ABMIL_MutantPriority
  /workspace/TCGA-LUAD/CLAM_SB_MutantPriority_Final

Output default:
  /workspace/TCGA-LUAD/ABMIL_vs_CLAM_Comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "auroc", "auprc", "sensitivity", "specificity", "accuracy",
    "balanced_accuracy", "precision", "npv", "f1", "mcc"
]

NAMES = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision / PPV",
    "npv": "NPV",
    "f1": "F1-score",
    "mcc": "MCC",
}


def read_json(path):
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")
    with open(path, "r") as f:
        return json.load(f)


def read_summary(model_dir):
    for name in ("cv_summary_threshold_0.5.csv", "cv_summary.csv"):
        path = model_dir / name
        if path.exists():
            return pd.read_csv(path), path
    raise SystemExit(f"Summary CV tidak ditemukan di: {model_dir}")


def read_oof(model_dir):
    path = model_dir / "oof_predictions.csv"
    if not path.exists():
        raise SystemExit(f"OOF tidak ditemukan: {path}")
    return pd.read_csv(path), path


def oof_row(model, agg):
    m = agg["oof_metrics_mutant_priority_threshold_development"]
    ci = agg.get("oof_auroc_ci95", [np.nan, np.nan])

    row = {
        "model": model,
        "n_cv": int(agg.get("n_cv_slides", 0)),
        "threshold": float(agg["selected_oof_threshold_development"]),
        "auroc_ci_low": float(ci[0]),
        "auroc_ci_high": float(ci[1]),
    }

    for metric in METRICS:
        row[metric] = float(m.get(metric, np.nan))

    for metric in ("tp", "tn", "fp", "fn"):
        row[metric] = int(m.get(metric, 0))

    return row


def validate_pairing(ab, cl):
    req = {"slide", "target"}
    for name, df in (("ABMIL", ab), ("CLAM", cl)):
        miss = req - set(df.columns)
        if miss:
            raise SystemExit(f"{name} OOF kekurangan kolom: {sorted(miss)}")

    a_cols = ["slide", "target"] + (["fold"] if "fold" in ab.columns else [])
    c_cols = ["slide", "target"] + (["fold"] if "fold" in cl.columns else [])

    a = ab[a_cols].rename(columns={"target": "target_ab", "fold": "fold_ab"})
    c = cl[c_cols].rename(columns={"target": "target_cl", "fold": "fold_cl"})

    x = a.merge(c, on="slide", how="outer", indicator=True)

    same_slides = bool((x["_merge"] == "both").all()) and len(x) == len(ab) == len(cl)
    same_target = same_slides and bool((x["target_ab"] == x["target_cl"]).all())

    same_fold = None
    if "fold_ab" in x.columns and "fold_cl" in x.columns and same_slides:
        same_fold = bool((x["fold_ab"] == x["fold_cl"]).all())

    return {
        "n_abmil": int(len(ab)),
        "n_clam": int(len(cl)),
        "same_slides": same_slides,
        "same_targets": same_target,
        "same_folds": same_fold,
        "paired_ready": bool(same_slides and same_target and same_fold is not False),
    }


def make_fold_table(ab, cl):
    cols = [
        "fold", "auroc", "auprc", "sensitivity", "specificity",
        "accuracy", "balanced_accuracy", "f1", "mcc"
    ]
    a = ab[cols].copy().rename(
        columns={c: f"abmil_{c}" for c in cols if c != "fold"}
    )
    c = cl[cols].copy().rename(
        columns={c: f"clam_{c}" for c in cols if c != "fold"}
    )
    out = a.merge(c, on="fold", validate="one_to_one").sort_values("fold")

    for metric in cols[1:]:
        out[f"delta_{metric}_clam_minus_abmil"] = (
            out[f"clam_{metric}"] - out[f"abmil_{metric}"]
        )
    return out


def make_mean_std(ab_agg, cl_agg):
    rows = []
    for metric in METRICS:
        am = float(ab_agg.get(f"{metric}_mean_at_0.5", np.nan))
        ast = float(ab_agg.get(f"{metric}_std_at_0.5", np.nan))
        cm = float(cl_agg.get(f"{metric}_mean_at_0.5", np.nan))
        cst = float(cl_agg.get(f"{metric}_std_at_0.5", np.nan))
        rows.append({
            "metric": NAMES[metric],
            "abmil_mean": am,
            "abmil_std": ast,
            "clam_mean": cm,
            "clam_std": cst,
            "delta_mean_clam_minus_abmil": cm - am,
        })
    return pd.DataFrame(rows)


def make_markdown(oof, mean_std, folds, paired):
    ab = oof[oof["model"] == "ABMIL"].iloc[0]
    cl = oof[oof["model"] == "CLAM"].iloc[0]

    lines = [
        "# Perbandingan Swin-Tiny + ABMIL vs Swin-Tiny + CLAM",
        "",
        "## OOF pada threshold Mutant-Priority",
        "",
        "| Metrik | ABMIL | CLAM | Δ CLAM−ABMIL |",
        "|---|---:|---:|---:|",
    ]

    for metric in METRICS:
        av, cv = ab[metric], cl[metric]
        lines.append(
            f"| {NAMES[metric]} | {av:.4f} | {cv:.4f} | {cv-av:+.4f} |"
        )

    lines += [
        f"| Threshold development | {ab['threshold']:.6f} | {cl['threshold']:.6f} | "
        f"{cl['threshold']-ab['threshold']:+.6f} |",
        f"| AUROC 95% CI | {ab['auroc_ci_low']:.3f}–{ab['auroc_ci_high']:.3f} | "
        f"{cl['auroc_ci_low']:.3f}–{cl['auroc_ci_high']:.3f} | — |",
        "",
        "## Mean ± SD 5-fold pada threshold 0.5",
        "",
        "| Metrik | ABMIL | CLAM | Δ mean CLAM−ABMIL |",
        "|---|---:|---:|---:|",
    ]

    for _, r in mean_std.iterrows():
        lines.append(
            f"| {r['metric']} | {r['abmil_mean']:.4f} ± {r['abmil_std']:.4f} | "
            f"{r['clam_mean']:.4f} ± {r['clam_std']:.4f} | "
            f"{r['delta_mean_clam_minus_abmil']:+.4f} |"
        )

    lines += [
        "",
        "## AUROC dan AUPRC per fold",
        "",
        "| Fold | ABMIL AUROC | CLAM AUROC | Δ AUROC | ABMIL AUPRC | CLAM AUPRC | Δ AUPRC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in folds.iterrows():
        lines.append(
            f"| {int(r['fold'])} | "
            f"{r['abmil_auroc']:.4f} | {r['clam_auroc']:.4f} | "
            f"{r['delta_auroc_clam_minus_abmil']:+.4f} | "
            f"{r['abmil_auprc']:.4f} | {r['clam_auprc']:.4f} | "
            f"{r['delta_auprc_clam_minus_abmil']:+.4f} |"
        )

    lines += [
        "",
        "## Validasi data paired",
        "",
        f"- Jumlah OOF ABMIL: {paired['n_abmil']}",
        f"- Jumlah OOF CLAM: {paired['n_clam']}",
        f"- Slide sama: {paired['same_slides']}",
        f"- Label sama: {paired['same_targets']}",
        f"- Fold sama: {paired['same_folds']}",
        f"- Siap untuk uji paired berikutnya: {paired['paired_ready']}",
        "",
        "> Catatan: tabel ini deskriptif. Signifikansi AUROC memerlukan uji DeLong, "
        "sedangkan hasil klasifikasi berpasangan dapat diuji dengan McNemar.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("/workspace/TCGA-LUAD"))
    ap.add_argument("--abmil-dir", default="ABMIL_MutantPriority")
    ap.add_argument("--clam-dir", default="CLAM_SB_MutantPriority_Final")
    ap.add_argument("--out-dir", default="ABMIL_vs_CLAM_Comparison")
    args = ap.parse_args()

    ab_dir = args.base_dir / args.abmil_dir
    cl_dir = args.base_dir / args.clam_dir
    out_dir = args.base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ab_agg = read_json(ab_dir / "cv_aggregate.json")
    cl_agg = read_json(cl_dir / "cv_aggregate.json")
    ab_sum, _ = read_summary(ab_dir)
    cl_sum, _ = read_summary(cl_dir)
    ab_oof, _ = read_oof(ab_dir)
    cl_oof, _ = read_oof(cl_dir)

    paired = validate_pairing(ab_oof, cl_oof)
    with open(out_dir / "oof_alignment_check.json", "w") as f:
        json.dump(paired, f, indent=2)

    if not paired["paired_ready"]:
        raise SystemExit("OOF ABMIL dan CLAM tidak identik. Jangan dibandingkan sebelum diperbaiki.")

    oof = pd.DataFrame([
        oof_row("ABMIL", ab_agg),
        oof_row("CLAM", cl_agg),
    ])
    mean_std = make_mean_std(ab_agg, cl_agg)
    folds = make_fold_table(ab_sum, cl_sum)

    oof.to_csv(out_dir / "comparison_oof.csv", index=False)
    mean_std.to_csv(out_dir / "comparison_fold_mean_std.csv", index=False)
    folds.to_csv(out_dir / "comparison_per_fold.csv", index=False)

    md = make_markdown(oof, mean_std, folds, paired)
    (out_dir / "comparison_table.md").write_text(md, encoding="utf-8")

    ab = oof[oof["model"] == "ABMIL"].iloc[0]
    cl = oof[oof["model"] == "CLAM"].iloc[0]

    rows = []
    for metric in METRICS:
        rows.append({
            "Metrik": NAMES[metric],
            "ABMIL": ab[metric],
            "CLAM": cl[metric],
            "Delta CLAM-ABMIL": cl[metric] - ab[metric],
        })
    table = pd.DataFrame(rows)

    print("=" * 95)
    print("PERBANDINGAN OOF ABMIL vs CLAM - MUTANT PRIORITY")
    print("=" * 95)
    print(table.to_string(
        index=False,
        formatters={
            "ABMIL": lambda x: f"{x:.4f}",
            "CLAM": lambda x: f"{x:.4f}",
            "Delta CLAM-ABMIL": lambda x: f"{x:+.4f}",
        }
    ))

    print()
    print(f"Threshold ABMIL : {ab['threshold']:.6f}")
    print(f"Threshold CLAM  : {cl['threshold']:.6f}")
    print(f"ABMIL AUROC 95% CI: {ab['auroc_ci_low']:.3f}-{ab['auroc_ci_high']:.3f}")
    print(f"CLAM  AUROC 95% CI: {cl['auroc_ci_low']:.3f}-{cl['auroc_ci_high']:.3f}")

    print("\nPER FOLD (AUROC / AUPRC)")
    print(folds[[
        "fold",
        "abmil_auroc", "clam_auroc", "delta_auroc_clam_minus_abmil",
        "abmil_auprc", "clam_auprc", "delta_auprc_clam_minus_abmil",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nVALIDASI PAIRED")
    print(json.dumps(paired, indent=2))

    print("\nOutput tersimpan di:")
    print(out_dir)
    print(" - comparison_oof.csv")
    print(" - comparison_fold_mean_std.csv")
    print(" - comparison_per_fold.csv")
    print(" - comparison_table.md")
    print(" - oof_alignment_check.json")
    print("=" * 95)


if __name__ == "__main__":
    main()
