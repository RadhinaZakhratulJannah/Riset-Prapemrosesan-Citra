#!/usr/bin/env python3
"""
FINAL TEST EVALUATION: Swin-Tiny + ABMIL vs Swin-Tiny + CLAM
TCGA-LUAD EGFR Mutant-Priority

Tujuan:
- Mengevaluasi kedua model pada TEST independen yang sebelumnya tidak disentuh.
- Menggunakan ensemble 5 checkpoint fold untuk masing-masing model.
- Menggunakan threshold Mutant-Priority yang SUDAH DIKUNCI dari OOF CV.
- TIDAK mencari threshold baru dari TEST.
- TIDAK melakukan DeLong/McNemar; program ini hanya menghasilkan prediksi dan metrik
  final test yang nantinya dapat dipakai untuk uji statistik final.

Default path:
  Base                  : /workspace/TCGA-LUAD
  Feature summary       : Features_Swin/features_summary.csv
  Feature tensors       : Features_Swin/pt_files
  ABMIL code            : /workspace/code/abmil_mutant_priority.py
  CLAM code             : /workspace/code/clam_mutant_priority_final.py
  ABMIL output/checkpts : ABMIL_MutantPriority
  CLAM output/checkpts  : CLAM_SB_MutantPriority_Final
  Output evaluasi       : Final_Test_ABMIL_vs_CLAM

PENTING:
Setelah program ini dijalankan dan label TEST dievaluasi, TEST dianggap sudah "dibuka".
Jangan mengubah hyperparameter, model, fold, atau threshold berdasarkan hasil TEST.
"""

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from concurrent.futures import ThreadPoolExecutor


def import_from_path(module_name: str, path: Path):
    if not path.exists():
        raise SystemExit(f"File kode tidak ditemukan: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Gagal membuat import spec untuk: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")
    with open(path, "r") as f:
        return json.load(f)


def encode_label(x):
    s = str(x).strip().lower()
    if s.startswith("mutant"):
        return 1
    if s.startswith("wildtype") or s in {"wt", "wild-type", "wild_type"}:
        return 0
    raise ValueError(f"Label tidak dikenali: {x}")


def get_pt_path(row, pt_dir: Path):
    p = str(row.get("pt", "")).strip()
    if p and p.lower() != "nan" and Path(p).exists():
        return Path(p)
    return pt_dir / f"{row['slide']}.pt"


def load_feature_half(path: Path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError(f"Feature harus tensor 2D: {path}")
    if x.shape[1] != 768:
        raise ValueError(f"Dimensi feature bukan 768: {path} -> {tuple(x.shape)}")
    return x.half().contiguous()


def build_test_cache(test_df, pt_dir: Path, workers=6):
    paths = [get_pt_path(row, pt_dir) for _, row in test_df.iterrows()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} file feature TEST tidak ditemukan. Contoh: {missing[:5]}"
        )

    print(f"Memuat feature TEST ({len(paths)} slide) ke RAM fp16 ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        tensors = list(ex.map(load_feature_half, paths))

    n_patches = int(sum(len(t) for t in tensors))
    total_gb = float(sum(t.numel() * 2 for t in tensors) / 1e9)
    print(
        f"TEST cache: {len(tensors)} slide | {n_patches:,} patch | {total_gb:.2f} GB fp16",
        flush=True,
    )
    return dict(zip(test_df["slide"].tolist(), tensors)), n_patches, total_gb


def binary_metrics(y_true, y_prob, threshold):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / max(len(y), 1)
    ppv = precision_score(y, pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = f1_score(y, pred, zero_division=0)
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0

    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if two else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "precision": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, n_boot=5000, seed=42):
    """Stratified bootstrap CI for AUROC and AUPRC."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    if len(pos) < 2 or len(neg) < 2:
        return {"auroc_ci95": [float("nan"), float("nan")],
                "auprc_ci95": [float("nan"), float("nan")]}

    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(int(n_boot)):
        idx_pos = rng.choice(pos, size=len(pos), replace=True)
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        yt = y[idx]
        pt = p[idx]
        aucs.append(roc_auc_score(yt, pt))
        aps.append(average_precision_score(yt, pt))

    return {
        "auroc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "auprc_ci95": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
    }


def validate_checkpoints(model_dir: Path, expected_arch: str):
    paths = []
    fold_meta = []
    for fold in range(1, 6):
        p = model_dir / f"fold_{fold}" / "best_model.pt"
        if not p.exists():
            raise SystemExit(f"Checkpoint fold {fold} tidak ditemukan: {p}")
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(p, map_location="cpu")
        arch = str(ck.get("arch", ""))
        if arch != expected_arch:
            raise SystemExit(
                f"Arsitektur checkpoint tidak sesuai: {p}\n"
                f"Diharapkan: {expected_arch}\nDitemukan: {arch}"
            )
        if int(ck.get("fold", fold)) != fold:
            raise SystemExit(f"Nomor fold checkpoint tidak sesuai: {p}")
        paths.append(p)
        fold_meta.append({
            "fold": fold,
            "epoch": int(ck.get("epoch", -1)),
            "checkpoint_score": float(ck.get("checkpoint_score", float("nan"))),
            "path": str(p),
        })
    return paths, fold_meta


def infer_ensemble(test_df, cache, ab_paths, cl_paths, abmod, clmod, device, amp=True):
    print("Memuat 5 checkpoint ABMIL + 5 checkpoint CLAM ...", flush=True)
    ab_models = []
    cl_models = []

    for p in ab_paths:
        ck = abmod.load_ckpt(p, device)
        ab_models.append(abmod.build_model_from_ckpt(ck, device))

    for p in cl_paths:
        ck = clmod.load_ckpt(p, device)
        cl_models.append(clmod.build_model_from_ckpt(ck, device))

    ab_probs = np.zeros((len(test_df), 5), dtype=np.float64)
    cl_probs = np.zeros((len(test_df), 5), dtype=np.float64)

    print("Menjalankan final inference pada TEST ...", flush=True)
    with torch.inference_mode():
        for i, row in test_df.iterrows():
            slide = row["slide"]
            x = cache[slide].to(device, non_blocking=True)

            # ABMIL ensemble
            for f, model in enumerate(ab_models):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(amp and device.type == "cuda"),
                ):
                    logit, _ = model(x)
                ab_probs[i, f] = float(torch.sigmoid(logit.float()).item())

            # CLAM ensemble
            for f, model in enumerate(cl_models):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(amp and device.type == "cuda"),
                ):
                    logits, _, _ = model(x)
                cl_probs[i, f] = float(torch.softmax(logits.float(), dim=1)[0, 1].item())

            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  [{i+1:02d}/{len(test_df):02d}] {slide}", flush=True)

    del ab_models, cl_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ab_probs, cl_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("/workspace/TCGA-LUAD"))
    ap.add_argument("--abmil-code", type=Path, default=Path("/workspace/code/abmil_mutant_priority.py"))
    ap.add_argument("--clam-code", type=Path, default=Path("/workspace/code/clam_mutant_priority_final.py"))
    ap.add_argument("--abmil-dir", default="ABMIL_MutantPriority")
    ap.add_argument("--clam-dir", default="CLAM_SB_MutantPriority_Final")
    ap.add_argument("--out-dir", default="Final_Test_ABMIL_vs_CLAM")
    ap.add_argument("--cache-workers", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/GPU tidak terdeteksi.")
    device = torch.device("cuda")
    amp = not args.no_amp

    base = args.base_dir
    feature_dir = base / "Features_Swin"
    summary_path = feature_dir / "features_summary.csv"
    pt_dir = feature_dir / "pt_files"
    ab_dir = base / args.abmil_dir
    cl_dir = base / args.clam_dir
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / "FINAL_TEST_OPENED_DO_NOT_TUNE.txt"
    if marker.exists() and not args.overwrite:
        raise SystemExit(
            f"Final TEST sebelumnya sudah dievaluasi: {marker}\n"
            "Program dihentikan agar tidak membuka TEST berulang kali.\n"
            "Gunakan --overwrite hanya jika memang perlu meregenerasi output yang sama, "
            "bukan untuk tuning model."
        )

    # Import exact model definitions from training scripts.
    abmod = import_from_path("abmil_mutant_priority_eval_source", args.abmil_code)
    clmod = import_from_path("clam_mutant_priority_eval_source", args.clam_code)

    if not summary_path.exists():
        raise SystemExit(f"Tidak ditemukan: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {"slide", "patient_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"features_summary.csv kekurangan kolom: {sorted(missing)}")

    split = df["split"].astype(str).str.lower().str.strip()
    train_df = df[split.eq("train")].copy().reset_index(drop=True)
    test_df = df[split.eq("test")].copy().reset_index(drop=True)

    if len(test_df) == 0:
        raise SystemExit("Tidak ada baris split=test pada features_summary.csv")

    train_df["target"] = train_df["label"].map(encode_label)
    test_df["target"] = test_df["label"].map(encode_label)

    for d in (train_df, test_df):
        d["patient_id"] = d["patient_id"].fillna("").astype(str)
        empty = d["patient_id"].str.strip().eq("")
        d.loc[empty, "patient_id"] = d.loc[empty, "slide"].astype(str)

    # Hard leakage checks.
    overlap_patients = sorted(set(train_df["patient_id"]) & set(test_df["patient_id"]))
    overlap_slides = sorted(set(train_df["slide"]) & set(test_df["slide"]))
    if overlap_patients:
        raise SystemExit(f"LEAKAGE: patient train-test overlap: {overlap_patients[:10]}")
    if overlap_slides:
        raise SystemExit(f"LEAKAGE: slide train-test overlap: {overlap_slides[:10]}")
    if test_df["slide"].duplicated().any():
        raise SystemExit("Ada slide TEST duplikat.")
    if test_df["patient_id"].duplicated().any():
        raise SystemExit("Ada patient TEST duplikat; expected 1 WSI per patient.")
    if test_df["target"].nunique() != 2:
        raise SystemExit("TEST tidak memiliki dua kelas.")

    # Locked thresholds from CV aggregate; never learned from TEST.
    ab_agg = read_json(ab_dir / "cv_aggregate.json")
    cl_agg = read_json(cl_dir / "cv_aggregate.json")
    thr_ab = float(ab_agg["selected_oof_threshold_development"])
    thr_cl = float(cl_agg["selected_oof_threshold_development"])

    ab_paths, ab_meta = validate_checkpoints(ab_dir, "gated_abmil_mutant_priority")
    cl_paths, cl_meta = validate_checkpoints(cl_dir, "clam_mutant_priority")

    print("=" * 105)
    print("FINAL TEST EVALUATION - ABMIL vs CLAM")
    print("=" * 105)
    print(f"GPU                    : {torch.cuda.get_device_name(0)}")
    print(f"TRAIN (hanya cek split): {len(train_df)}")
    print(f"TEST final             : {len(test_df)} | {test_df['target'].value_counts().sort_index().to_dict()}")
    print(f"ABMIL threshold locked : {thr_ab:.6f}")
    print(f"CLAM threshold locked  : {thr_cl:.6f}")
    print("Threshold TIDAK akan diubah berdasarkan TEST.")
    print("=" * 105)

    cache, n_patches, cache_gb = build_test_cache(test_df, pt_dir, args.cache_workers)

    ab_probs, cl_probs = infer_ensemble(
        test_df, cache, ab_paths, cl_paths, abmod, clmod, device, amp=amp
    )

    y = test_df["target"].to_numpy(dtype=int)
    ab_mean = ab_probs.mean(axis=1)
    cl_mean = cl_probs.mean(axis=1)
    ab_std = ab_probs.std(axis=1)
    cl_std = cl_probs.std(axis=1)

    # Final locked-threshold metrics.
    ab_m = binary_metrics(y, ab_mean, thr_ab)
    cl_m = binary_metrics(y, cl_mean, thr_cl)
    ab_ci = bootstrap_ci(y, ab_mean, args.bootstrap, args.seed)
    cl_ci = bootstrap_ci(y, cl_mean, args.bootstrap, args.seed + 1)

    # Reference metrics at 0.5 only; not used to retune.
    ab_m05 = binary_metrics(y, ab_mean, 0.5)
    cl_m05 = binary_metrics(y, cl_mean, 0.5)

    pred = test_df[["slide", "patient_id", "label", "target"]].copy()
    for f in range(5):
        pred[f"abmil_prob_fold_{f+1}"] = ab_probs[:, f]
        pred[f"clam_prob_fold_{f+1}"] = cl_probs[:, f]
    pred["abmil_prob_mutant"] = ab_mean
    pred["abmil_prob_std"] = ab_std
    pred["abmil_threshold_locked"] = thr_ab
    pred["abmil_pred_mutant"] = (ab_mean >= thr_ab).astype(int)
    pred["abmil_correct"] = (pred["abmil_pred_mutant"].to_numpy() == y).astype(int)
    pred["clam_prob_mutant"] = cl_mean
    pred["clam_prob_std"] = cl_std
    pred["clam_threshold_locked"] = thr_cl
    pred["clam_pred_mutant"] = (cl_mean >= thr_cl).astype(int)
    pred["clam_correct"] = (pred["clam_pred_mutant"].to_numpy() == y).astype(int)
    pred.to_csv(out_dir / "final_test_paired_predictions.csv", index=False)

    metrics_payload = {
        "analysis": "independent_final_test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_split_check_only": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_test_mutant": int((y == 1).sum()),
        "n_test_wildtype": int((y == 0).sum()),
        "test_patches": int(n_patches),
        "test_cache_fp16_gb": float(cache_gb),
        "thresholds_locked_from_oof_cv": {
            "ABMIL": thr_ab,
            "CLAM": thr_cl,
        },
        "ABMIL": {
            "locked_threshold_metrics": ab_m,
            "bootstrap_ci95": ab_ci,
            "reference_threshold_0.5_metrics": ab_m05,
            "checkpoint_metadata": ab_meta,
        },
        "CLAM": {
            "locked_threshold_metrics": cl_m,
            "bootstrap_ci95": cl_ci,
            "reference_threshold_0.5_metrics": cl_m05,
            "checkpoint_metadata": cl_meta,
        },
        "warning": (
            "FINAL TEST has been opened. Do not tune hyperparameters, thresholds, folds, "
            "or model selection based on these results."
        ),
    }
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    metric_order = [
        "auroc", "auprc", "sensitivity", "specificity", "accuracy",
        "balanced_accuracy", "precision", "npv", "f1", "mcc",
    ]
    rows = []
    for metric in metric_order:
        rows.append({
            "metric": metric,
            "ABMIL": ab_m[metric],
            "CLAM": cl_m[metric],
            "delta_CLAM_minus_ABMIL": cl_m[metric] - ab_m[metric],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "final_test_metric_comparison.csv", index=False)

    # Human-readable markdown.
    md = []
    md.append("# Final Test Evaluation — ABMIL vs CLAM")
    md.append("")
    md.append(f"- TEST: **{len(test_df)} WSI** ({int((y==1).sum())} Mutant, {int((y==0).sum())} Wild-Type)")
    md.append(f"- Locked ABMIL threshold: **{thr_ab:.6f}**")
    md.append(f"- Locked CLAM threshold: **{thr_cl:.6f}**")
    md.append("- Probabilitas final: mean ensemble dari 5 checkpoint fold.")
    md.append("")
    md.append("| Metrik | ABMIL | CLAM | Δ CLAM−ABMIL |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['metric']} | {r['ABMIL']:.4f} | {r['CLAM']:.4f} | "
            f"{r['delta_CLAM_minus_ABMIL']:+.4f} |"
        )
    md.append("")
    md.append(
        f"ABMIL AUROC 95% bootstrap CI: **{ab_ci['auroc_ci95'][0]:.3f}–{ab_ci['auroc_ci95'][1]:.3f}**"
    )
    md.append(
        f"CLAM AUROC 95% bootstrap CI: **{cl_ci['auroc_ci95'][0]:.3f}–{cl_ci['auroc_ci95'][1]:.3f}**"
    )
    md.append("")
    md.append(
        "> TEST sudah dibuka. Jangan melakukan tuning model/threshold berdasarkan hasil ini. "
        "Gunakan final_test_paired_predictions.csv untuk DeLong dan McNemar final."
    )
    (out_dir / "final_test_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    marker.write_text(
        "FINAL TEST OPENED\n"
        f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"ABMIL threshold locked from OOF: {thr_ab:.9f}\n"
        f"CLAM threshold locked from OOF: {thr_cl:.9f}\n"
        "Do not tune models or thresholds based on final test results.\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 105)
    print("HASIL FINAL TEST - LOCKED THRESHOLDS")
    print("=" * 105)
    print(f"{'Metric':22s} {'ABMIL':>12s} {'CLAM':>12s} {'CLAM-ABMIL':>14s}")
    for r in rows:
        print(
            f"{r['metric']:22s} {r['ABMIL']:12.4f} {r['CLAM']:12.4f} "
            f"{r['delta_CLAM_minus_ABMIL']:+14.4f}"
        )

    print("\nConfusion matrix counts @ locked threshold")
    print(f"ABMIL: TP={ab_m['tp']} TN={ab_m['tn']} FP={ab_m['fp']} FN={ab_m['fn']}")
    print(f"CLAM : TP={cl_m['tp']} TN={cl_m['tn']} FP={cl_m['fp']} FN={cl_m['fn']}")
    print(
        f"\nABMIL AUROC 95% bootstrap CI: {ab_ci['auroc_ci95'][0]:.3f}-{ab_ci['auroc_ci95'][1]:.3f}"
    )
    print(
        f"CLAM  AUROC 95% bootstrap CI: {cl_ci['auroc_ci95'][0]:.3f}-{cl_ci['auroc_ci95'][1]:.3f}"
    )
    print("\nOutput:")
    print(f"  {out_dir / 'final_test_paired_predictions.csv'}")
    print(f"  {out_dir / 'final_test_metrics.json'}")
    print(f"  {out_dir / 'final_test_metric_comparison.csv'}")
    print(f"  {out_dir / 'final_test_summary.md'}")
    print(f"  {marker}")
    print("\nTEST SUDAH DIBUKA. Jangan tuning lagi setelah melihat hasil ini.")
    print("=" * 105)


if __name__ == "__main__":
    main()
