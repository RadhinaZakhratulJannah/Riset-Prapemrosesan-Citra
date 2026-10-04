"""
ABMIL Mutant-Priority 5-Fold CV untuk Swin-Tiny TCGA-LUAD EGFR.

Fokus:
- Kelas positif = EGFR Mutant (target=1).
- Balanced bag oversampling: mutant diulang agar jumlah presentasi bag mutant
  per epoch mendekati wild-type, tanpa membuang wild-type.
- Positive loss weight moderat (default 2.0) untuk menekan false negative.
- Checkpoint dipilih dari skor gabungan AUROC + AUPRC, bukan accuracy.
- Setelah 5-fold selesai, threshold OOF dipilih dengan syarat sensitivity
  development >= target (default 0.80), lalu specificity dimaksimalkan.
- TEST tidak disentuh kecuali --evaluate-test diberikan secara eksplisit.
- Fold disimpan di cv_folds_mutant_priority.csv dan dipakai ulang oleh CLAM.

Catatan metodologis:
Threshold target-sensitivity yang dihitung dari OOF adalah threshold DEVELOPMENT.
Jangan memakai hasil TEST untuk tuning. Evaluasi TEST hanya sekali setelah
hyperparameter dikunci.

Contoh CV:
nohup python /workspace/code/abmil_mutant_priority.py \
  --out-name ABMIL_MutantPriority \
  --epochs 40 --patience 10 \
  --lr 0.0001 --weight-decay 0.0001 \
  --dropout 0.35 --pos-weight 2.0 \
  --max-instances-train 15000 --grad-accum 4 \
  --target-sensitivity 0.80 \
  > /workspace/TCGA-LUAD/abmil_mutant_priority.log 2>&1 &

Evaluasi TEST setelah konfigurasi final dikunci:
python /workspace/code/abmil_mutant_priority.py \
  --out-name ABMIL_MutantPriority --evaluate-test
"""

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


# -----------------------------------------------------------------------------
# Reproducibility / label helpers
# -----------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_int(text: str) -> int:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def encode_label(x):
    s = str(x).strip().lower()
    if s.startswith("mutant"):
        return 1
    if s.startswith("wildtype") or s in {"wt", "wild-type", "wild_type"}:
        return 0
    raise ValueError(f"Label tidak dikenali: {x}")


def extract_tss(patient_id, slide):
    for value in (patient_id, slide):
        parts = str(value).strip().split("-")
        if len(parts) >= 2 and parts[0].upper() == "TCGA":
            return parts[1].upper()
    return "UNK"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GatedABMIL(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, attn_dim=128, dropout=0.35):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout * 0.5),
        )
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        h = self.embed(x)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(1)
        a = torch.softmax(a.float(), dim=0).to(h.dtype)
        m = torch.sum(a.unsqueeze(1) * h, dim=0)
        logit = self.classifier(m).squeeze(0)
        return logit, a


# -----------------------------------------------------------------------------
# Metrics / operating threshold
# -----------------------------------------------------------------------------
def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    two_classes = len(np.unique(y_true)) == 2
    auroc = float(roc_auc_score(y_true, y_prob)) if two_classes else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if two_classes else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    precision = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

    return {
        "threshold": float(threshold),
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def threshold_for_target_sensitivity(y_true, y_prob, target_sensitivity=0.80):
    """Pilih specificity tertinggi dengan sensitivity >= target pada data development."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    if len(thresholds) == 0:
        return 0.5, {"target_met": False, "reason": "no_finite_threshold"}

    specificity = 1.0 - fpr
    eligible = np.where(tpr + 1e-12 >= float(target_sensitivity))[0]

    if len(eligible):
        best_spec = np.max(specificity[eligible])
        candidates = eligible[np.isclose(specificity[eligible], best_spec)]
        # Di antara specificity yang sama, ambil threshold paling tinggi agar FP tidak berlebihan.
        best_idx = candidates[np.argmax(thresholds[candidates])]
        target_met = True
    else:
        # Secara teoritis ROC biasanya punya titik sens=1, tetapi sediakan fallback aman.
        best_tpr = np.max(tpr)
        candidates = np.where(np.isclose(tpr, best_tpr))[0]
        best_spec = np.max(specificity[candidates])
        candidates = candidates[np.isclose(specificity[candidates], best_spec)]
        best_idx = candidates[np.argmax(thresholds[candidates])]
        target_met = False

    return float(thresholds[best_idx]), {
        "target_met": bool(target_met),
        "target_sensitivity": float(target_sensitivity),
        "development_sensitivity_at_selected_point": float(tpr[best_idx]),
        "development_specificity_at_selected_point": float(specificity[best_idx]),
    }


def bootstrap_auc_ci(y, p, n=2000, seed=42):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n)):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def checkpoint_score(metrics):
    """Ranking quality with extra attention to positive class through AUPRC."""
    auc = metrics.get("auroc", float("nan"))
    ap = metrics.get("auprc", float("nan"))
    if not np.isfinite(auc):
        return -math.inf
    if not np.isfinite(ap):
        ap = 0.0
    return 0.5 * float(auc) + 0.5 * float(ap)


# -----------------------------------------------------------------------------
# Feature loading / sampling
# -----------------------------------------------------------------------------
def load_tensor(path: Path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError(f"Feature tidak valid: {path}")
    return x.float()


def get_pt_path(row, pt_dir: Path):
    p = str(row.get("pt", "")).strip()
    if p and p.lower() != "nan" and Path(p).exists():
        return Path(p)
    return pt_dir / f"{row['slide']}.pt"


def maybe_sample_instances(x, max_instances, seed):
    if max_instances <= 0 or len(x) <= max_instances:
        return x
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(x), generator=g)[:max_instances]
    return x[idx]


def make_epoch_indices(frame, rng, balanced_bags=True):
    """
    Jika balanced_bags=True:
      - semua WT dipakai sekali
      - Mutant di-oversample dengan replacement sampai sebanyak WT
    Tidak ada data validation/test yang masuk ke proses ini.
    """
    all_idx = np.arange(len(frame))
    if not balanced_bags:
        return rng.permutation(all_idx)

    targets = frame["target"].to_numpy().astype(int)
    neg = all_idx[targets == 0]
    pos = all_idx[targets == 1]
    if len(neg) == 0 or len(pos) == 0:
        return rng.permutation(all_idx)

    if len(pos) < len(neg):
        pos_epoch = rng.choice(pos, size=len(neg), replace=True)
        neg_epoch = neg.copy()
    elif len(neg) < len(pos):
        neg_epoch = rng.choice(neg, size=len(pos), replace=True)
        pos_epoch = pos.copy()
    else:
        neg_epoch, pos_epoch = neg.copy(), pos.copy()

    order = np.concatenate([neg_epoch, pos_epoch])
    return rng.permutation(order)


# -----------------------------------------------------------------------------
# Fair fixed folds for ABMIL + CLAM
# -----------------------------------------------------------------------------
def fold_balance_score(frame, fold_arr):
    global_pos = float(frame["target"].mean())
    global_tss = frame["tss"].value_counts(normalize=True)
    target_size = len(frame) / 5.0
    score = 0.0
    for fold in range(1, 6):
        g = frame[fold_arr == fold]
        if len(g) == 0 or g["target"].nunique() < 2:
            return float("inf")
        class_pen = abs(float(g["target"].mean()) - global_pos)
        size_pen = abs(len(g) - target_size) / target_size
        dist = g["tss"].value_counts(normalize=True)
        idx = global_tss.index.union(dist.index)
        tss_pen = float(
            (global_tss.reindex(idx, fill_value=0) - dist.reindex(idx, fill_value=0)).abs().sum() / 2.0
        )
        score += 5.0 * class_pen + 0.5 * size_pen + 0.5 * tss_pen
    return score


def build_or_load_folds(train_df, fold_path: Path, seed: int, trials: int):
    if fold_path.exists():
        f = pd.read_csv(fold_path)
        merged = train_df[["slide"]].merge(f[["slide", "fold"]], on="slide", how="left")
        if len(f) != len(train_df) or merged["fold"].isna().any():
            raise SystemExit(
                f"{fold_path} tidak cocok dengan TRAIN saat ini. "
                "Hapus fold file hanya jika dataset memang berubah."
            )
        print(f"Fold dimuat dari {fold_path}")
        return merged["fold"].astype(int).to_numpy()

    best = None
    best_seed = None
    best_score = float("inf")
    for cand_seed in range(int(seed), int(seed) + max(int(trials), 1)):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cand_seed)
        arr = np.zeros(len(train_df), dtype=int)
        ok = True
        for k, (_, va) in enumerate(
            splitter.split(train_df, train_df["target"], groups=train_df["patient_id"]), start=1
        ):
            arr[va] = k
            if train_df.iloc[va]["target"].nunique() < 2:
                ok = False
        if not ok:
            continue
        s = fold_balance_score(train_df, arr)
        if s < best_score:
            best_score, best_seed, best = s, cand_seed, arr.copy()

    if best is None:
        raise SystemExit("Gagal membuat 5-fold yang valid.")

    out = train_df[["slide", "patient_id", "label", "target", "tss"]].copy()
    out["fold"] = best
    out.to_csv(fold_path, index=False)
    with open(fold_path.with_suffix(".json"), "w") as f:
        json.dump(
            {"chosen_seed": int(best_seed), "balance_score": float(best_score), "trials": int(trials)},
            f,
            indent=2,
        )
    print(f"Fold baru disimpan: {fold_path} | chosen_seed={best_seed} | balance_score={best_score:.4f}")
    return best


# -----------------------------------------------------------------------------
# Train / predict
# -----------------------------------------------------------------------------
def train_epoch(
    model,
    frame,
    pt_dir,
    device,
    optimizer,
    criterion,
    scaler,
    epoch,
    fold,
    base_seed,
    max_instances_train,
    amp,
    grad_clip,
    grad_accum,
    balanced_bags,
):
    model.train()
    rng = np.random.default_rng(base_seed + fold * 1000 + epoch)
    order = make_epoch_indices(frame, rng, balanced_bags=balanced_bags)
    losses, y_true, y_prob = [], [], []
    optimizer.zero_grad(set_to_none=True)
    grad_accum = max(int(grad_accum), 1)

    for step, idx in enumerate(order, start=1):
        row = frame.iloc[int(idx)]
        x = load_tensor(get_pt_path(row, pt_dir))
        sample_seed = (
            base_seed
            + fold * 1000003
            + epoch * 10007
            + step * 101
            + stable_int(row["slide"]) % 100000
        )
        x = maybe_sample_instances(x, max_instances_train, sample_seed)
        y = float(row["target"])
        x = x.to(device, non_blocking=True)
        target = torch.tensor(y, device=device, dtype=torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logit, _ = model(x)
            loss = criterion(logit.float(), target)

        scaler.scale(loss / grad_accum).backward()
        do_step = (step % grad_accum == 0) or (step == len(order))
        if do_step:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        prob = torch.sigmoid(logit.detach().float()).item()
        losses.append(float(loss.item()))
        y_true.append(int(y))
        y_prob.append(prob)
        del x, target, logit, loss

    m = binary_metrics(y_true, y_prob, 0.5)
    m["loss"] = float(np.mean(losses))
    m["n_bag_presentations"] = int(len(order))
    return m


@torch.inference_mode()
def predict_frame(model, frame, pt_dir, device, amp, criterion=None):
    model.eval()
    rows, losses = [], []
    for _, row in frame.iterrows():
        x = load_tensor(get_pt_path(row, pt_dir)).to(device, non_blocking=True)
        target = torch.tensor(float(row["target"]), device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logit, _ = model(x)
        if criterion is not None:
            losses.append(float(criterion(logit.float(), target).item()))
        prob = torch.sigmoid(logit.float()).item()
        rows.append(
            {
                "slide": row["slide"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                "target": int(row["target"]),
                "prob_mutant": prob,
            }
        )
        del x, target, logit
    pred = pd.DataFrame(rows)
    return pred, (float(np.mean(losses)) if losses else float("nan"))


def build_model_from_ckpt(ckpt, device):
    model = GatedABMIL(
        input_dim=int(ckpt.get("input_dim", 768)),
        embed_dim=int(ckpt["embed_dim"]),
        attn_dim=int(ckpt["attn_dim"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("/workspace/TCGA-LUAD"))
    ap.add_argument("--out-name", default="ABMIL_MutantPriority")
    ap.add_argument("--fold-file", default="cv_folds_mutant_priority.csv")
    ap.add_argument("--fold-search-trials", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--attn-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--max-instances-train", type=int, default=15000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--scheduler-patience", type=int, default=3)
    ap.add_argument("--target-sensitivity", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-balanced-bags", action="store_true")
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.target_sensitivity <= 1.0):
        raise SystemExit("--target-sensitivity harus >0 dan <=1.")
    if args.pos_weight <= 0:
        raise SystemExit("--pos-weight harus >0.")

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/GPU tidak terdeteksi.")
    device = torch.device("cuda")
    amp = not args.no_amp
    balanced_bags = not args.no_balanced_bags

    feature_dir = args.base_dir / "Features_Swin"
    summary_csv = feature_dir / "features_summary.csv"
    pt_dir = feature_dir / "pt_files"
    out_dir = args.base_dir / args.out_name
    fold_path = args.base_dir / args.fold_file
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise SystemExit(f"Tidak ditemukan: {summary_csv}")

    df = pd.read_csv(summary_csv)
    required = {"slide", "patient_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Kolom features_summary kurang: {missing}")

    split = df["split"].astype(str).str.lower()
    train_df = df[split.eq("train")].copy().reset_index(drop=True)
    test_df = df[split.eq("test")].copy().reset_index(drop=True)
    train_df["target"] = train_df["label"].map(encode_label)
    if len(test_df):
        test_df["target"] = test_df["label"].map(encode_label)

    for d in (train_df, test_df):
        d["patient_id"] = d["patient_id"].fillna("").astype(str)
        empty = d["patient_id"].str.strip().eq("")
        d.loc[empty, "patient_id"] = d.loc[empty, "slide"]
        d["tss"] = [extract_tss(p, s) for p, s in zip(d["patient_id"], d["slide"])]

    if train_df["slide"].duplicated().any():
        raise SystemExit("Ada slide duplikat di TRAIN.")
    shared = set(train_df["patient_id"]) & set(test_df["patient_id"])
    if shared:
        raise SystemExit(f"KEBOCORAN: {len(shared)} pasien muncul di TRAIN dan TEST.")
    groups_per_class = train_df.groupby("target")["patient_id"].nunique()
    if len(groups_per_class) < 2 or groups_per_class.min() < 5:
        raise SystemExit("Setiap kelas membutuhkan minimal 5 patient untuk 5-fold.")

    missing_pt = [
        str(get_pt_path(r, pt_dir))
        for _, r in train_df.iterrows()
        if not get_pt_path(r, pt_dir).exists()
    ]
    if missing_pt:
        raise SystemExit(f"{len(missing_pt)} file .pt TRAIN hilang. Contoh: {missing_pt[:3]}")

    signature = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "pos_weight": args.pos_weight,
        "max_instances_train": args.max_instances_train,
        "grad_clip": args.grad_clip,
        "grad_accum": args.grad_accum,
        "scheduler_patience": args.scheduler_patience,
        "target_sensitivity": args.target_sensitivity,
        "seed": args.seed,
        "balanced_bags": balanced_bags,
        "fold_file": args.fold_file,
    }
    sig_path = out_dir / "training_signature.json"
    if sig_path.exists():
        old = json.load(open(sig_path))
        if old != signature:
            raise SystemExit(
                f"Konfigurasi berbeda dari run yang sudah ada di {out_dir}. "
                "Gunakan --out-name baru atau hapus folder output lama sebelum training ulang."
            )
    else:
        with open(sig_path, "w") as f:
            json.dump(signature, f, indent=2)

    config = vars(args).copy()
    config["base_dir"] = str(config["base_dir"])
    config["balanced_bags"] = balanced_bags
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    fold_arr = build_or_load_folds(train_df, fold_path, args.seed, args.fold_search_trials)

    print("=" * 100)
    print("ABMIL MUTANT-PRIORITY 5-FOLD")
    print("=" * 100)
    print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"TRAIN untuk CV    : {len(train_df)} | {train_df['target'].value_counts().sort_index().to_dict()}")
    print(f"TEST disimpan     : {len(test_df)} (tidak disentuh saat tuning)")
    print(f"Balanced bags     : {balanced_bags}")
    print(f"Positive weight   : {args.pos_weight}")
    print(f"Target sensitivity: {args.target_sensitivity:.2f} (OOF development threshold)")
    print(f"Fold file         : {fold_path}")
    print(f"Output            : {out_dir}")

    train_df.to_csv(out_dir / "cv_train_dataset.csv", index=False)
    test_df.to_csv(out_dir / "untouched_test_dataset.csv", index=False)

    fold_results, oof_frames, model_paths = [], [], []

    for fold in range(1, 6):
        seed_everything(args.seed + fold * 1000)
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_path = fold_dir / "best_model.pt"
        val_pred_path = fold_dir / "val_predictions.csv"
        metrics_path = fold_dir / "metrics_threshold_0.5.json"
        model_paths.append(best_path)

        tr = train_df[fold_arr != fold].copy().reset_index(drop=True)
        va = train_df[fold_arr == fold].copy().reset_index(drop=True)
        tr.to_csv(fold_dir / "train_split.csv", index=False)
        va.to_csv(fold_dir / "val_split.csv", index=False)

        if set(tr["patient_id"]) & set(va["patient_id"]):
            raise RuntimeError(f"Patient leakage fold {fold}")
        if tr["target"].nunique() < 2 or va["target"].nunique() < 2:
            raise RuntimeError(f"Fold {fold} tidak mempunyai dua kelas.")

        print("\n" + "=" * 100)
        print(
            f"FOLD {fold}/5 | train={len(tr)} {tr['target'].value_counts().sort_index().to_dict()} "
            f"| val={len(va)} {va['target'].value_counts().sort_index().to_dict()}"
        )
        print("=" * 100)

        if best_path.exists() and val_pred_path.exists() and metrics_path.exists():
            print(f"Fold {fold} sudah selesai -> SKIP training")
            pred = pd.read_csv(val_pred_path)
            m = binary_metrics(pred["target"], pred["prob_mutant"], 0.5)
            m.update(
                {
                    "fold": fold,
                    "best_epoch": int(pred.get("best_epoch", pd.Series([0])).iloc[0]),
                    "n_train": len(tr),
                    "n_val": len(va),
                }
            )
            fold_results.append(m)
            pred["fold"] = fold
            oof_frames.append(pred)
            continue

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(args.pos_weight), device=device, dtype=torch.float32)
        )
        model = GatedABMIL(768, args.embed_dim, args.attn_dim, args.dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1e-6,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        best_score = -math.inf
        best_epoch = 0
        bad_epochs = 0
        history = []
        fold_start = time.time()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_m = train_epoch(
                model,
                tr,
                pt_dir,
                device,
                optimizer,
                criterion,
                scaler,
                epoch,
                fold,
                args.seed,
                args.max_instances_train,
                amp,
                args.grad_clip,
                args.grad_accum,
                balanced_bags,
            )
            val_pred, val_loss = predict_frame(model, va, pt_dir, device, amp, criterion)
            val_m = binary_metrics(val_pred["target"], val_pred["prob_mutant"], 0.5)
            val_m["loss"] = val_loss
            score = checkpoint_score(val_m)
            scheduler.step(score if np.isfinite(score) else 0.0)
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "checkpoint_score": score,
                "lr": current_lr,
                "train_loss": train_m["loss"],
                "train_auroc": train_m["auroc"],
                "train_auprc": train_m["auprc"],
                "val_loss": val_m["loss"],
                "val_auroc": val_m["auroc"],
                "val_auprc": val_m["auprc"],
                "val_sensitivity_at_0.5": val_m["sensitivity"],
                "val_specificity_at_0.5": val_m["specificity"],
                "val_accuracy_at_0.5": val_m["accuracy"],
                "val_balanced_accuracy_at_0.5": val_m["balanced_accuracy"],
                "val_f1_at_0.5": val_m["f1"],
                "val_mcc_at_0.5": val_m["mcc"],
                "n_bag_presentations": train_m["n_bag_presentations"],
                "epoch_sec": round(time.time() - t0, 1),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

            print(
                f"Fold {fold} | Ep {epoch:02d}/{args.epochs} | "
                f"train AUC {train_m['auroc']:.3f} AP {train_m['auprc']:.3f} | "
                f"val AUC {val_m['auroc']:.3f} AP {val_m['auprc']:.3f} score {score:.3f} | "
                f"sens@0.5 {val_m['sensitivity']:.3f} spec@0.5 {val_m['specificity']:.3f} "
                f"balacc {val_m['balanced_accuracy']:.3f} | lr {current_lr:.2e} | {row['epoch_sec']:.0f}s",
                flush=True,
            )

            if score > best_score + 1e-6:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                torch.save(
                    {
                        "arch": "gated_abmil_mutant_priority",
                        "fold": fold,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "input_dim": 768,
                        "embed_dim": args.embed_dim,
                        "attn_dim": args.attn_dim,
                        "dropout": args.dropout,
                        "val_metrics": val_m,
                        "checkpoint_score": score,
                        "pos_weight": float(args.pos_weight),
                        "balanced_bags": balanced_bags,
                        "seed": args.seed + fold * 1000,
                    },
                    best_path,
                )
                val_pred.to_csv(fold_dir / "best_epoch_predictions.csv", index=False)
            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                print(
                    f"Early stopping fold {fold}: best epoch={best_epoch}, "
                    f"best composite={best_score:.4f}"
                )
                break

        ckpt = load_ckpt(best_path, device)
        model = build_model_from_ckpt(ckpt, device)
        pred, _ = predict_frame(model, va, pt_dir, device, amp, None)
        pred["fold"] = fold
        pred["best_epoch"] = best_epoch
        pred.to_csv(val_pred_path, index=False)

        m = binary_metrics(pred["target"], pred["prob_mutant"], 0.5)
        result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "checkpoint_score": float(best_score),
            "n_train": len(tr),
            "n_val": len(va),
            "pos_weight": float(args.pos_weight),
            "balanced_bags": balanced_bags,
            **m,
            "fold_time_min": round((time.time() - fold_start) / 60, 2),
        }
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)
        fold_results.append(result)
        oof_frames.append(pred)
        print(
            f"BEST FOLD {fold}: AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | "
            f"Sens@0.5 {m['sensitivity']:.4f} | Spec@0.5 {m['specificity']:.4f}"
        )

        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(fold_results).sort_values("fold")
    summary.to_csv(out_dir / "cv_summary_threshold_0.5.csv", index=False)
    oof = pd.concat(oof_frames, ignore_index=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    threshold, threshold_info = threshold_for_target_sensitivity(
        oof["target"], oof["prob_mutant"], args.target_sensitivity
    )
    oof_m05 = binary_metrics(oof["target"], oof["prob_mutant"], 0.5)
    oof_mtarget = binary_metrics(oof["target"], oof["prob_mutant"], threshold)
    ci = bootstrap_auc_ci(oof["target"], oof["prob_mutant"], seed=args.seed)

    threshold_payload = {
        "method": "maximize_specificity_subject_to_minimum_sensitivity",
        "threshold": threshold,
        **threshold_info,
        "warning": "Development threshold dari seluruh OOF; kunci threshold ini sebelum evaluasi TEST.",
    }
    with open(out_dir / "mutant_priority_threshold.json", "w") as f:
        json.dump(threshold_payload, f, indent=2)

    metric_cols = [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "npv",
        "f1",
        "mcc",
    ]
    aggregate = {
        "checkpoint_monitor": "0.5*AUROC + 0.5*AUPRC",
        "target_sensitivity": float(args.target_sensitivity),
        "selected_oof_threshold_development": float(threshold),
        "threshold_info": threshold_info,
        "oof_auroc_ci95": ci,
        "oof_metrics_threshold_0.5": oof_m05,
        "oof_metrics_mutant_priority_threshold_development": oof_mtarget,
        "n_cv_slides": int(len(oof)),
        "n_test_untouched": int(len(test_df)),
    }
    for c in metric_cols:
        aggregate[f"{c}_mean_at_0.5"] = float(summary[c].mean())
        aggregate[f"{c}_std_at_0.5"] = float(summary[c].std(ddof=1))

    with open(out_dir / "cv_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 100)
    print("HASIL CV ABMIL MUTANT-PRIORITY")
    print("=" * 100)
    show = [
        "fold",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mcc",
    ]
    print(summary[show].to_string(index=False))
    print(f"\nOOF AUROC : {oof_m05['auroc']:.4f} | 95% CI {ci[0]:.3f}-{ci[1]:.3f}")
    print(f"OOF AUPRC : {oof_m05['auprc']:.4f}")
    print(
        f"Mutant-priority threshold (development): {threshold:.6f} "
        f"untuk target sensitivity >= {args.target_sensitivity:.2f}"
    )
    print(
        f"OOF @ mutant-priority -> sens {oof_mtarget['sensitivity']:.3f} | "
        f"spec {oof_mtarget['specificity']:.3f} | acc {oof_mtarget['accuracy']:.3f} | "
        f"balacc {oof_mtarget['balanced_accuracy']:.3f} | F1 {oof_mtarget['f1']:.3f} | "
        f"MCC {oof_mtarget['mcc']:.3f}"
    )

    if args.evaluate_test:
        missing_test_pt = [
            str(get_pt_path(r, pt_dir))
            for _, r in test_df.iterrows()
            if not get_pt_path(r, pt_dir).exists()
        ]
        if missing_test_pt:
            raise SystemExit(f"{len(missing_test_pt)} file .pt TEST hilang. Contoh: {missing_test_pt[:3]}")

        test_probs = []
        for model_path in model_paths:
            if not model_path.exists():
                raise SystemExit(f"Model fold belum lengkap: {model_path}")
            ckpt = load_ckpt(model_path, device)
            model = build_model_from_ckpt(ckpt, device)
            pred, _ = predict_frame(model, test_df, pt_dir, device, amp, None)
            test_probs.append(pred["prob_mutant"].to_numpy())
            del model
            torch.cuda.empty_cache()

        test_out = test_df[["slide", "patient_id", "label", "target"]].copy()
        probs = np.stack(test_probs, axis=1)
        for i in range(5):
            test_out[f"prob_fold_{i+1}"] = probs[:, i]
        test_out["prob_mutant"] = probs.mean(axis=1)
        test_out["prob_std"] = probs.std(axis=1)
        test_out["pred_mutant"] = (test_out["prob_mutant"] >= threshold).astype(int)
        test_out.to_csv(out_dir / "test_ensemble_predictions.csv", index=False)

        test_metrics = binary_metrics(test_out["target"], test_out["prob_mutant"], threshold)
        with open(out_dir / "test_ensemble_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        print("\nFINAL TEST 5-FOLD ENSEMBLE")
        print("Threshold sudah dikunci dari OOF CV sebelum TEST dibuka.")
        for k in metric_cols:
            print(f"  {k:18s}: {test_metrics[k]:.4f}")
    else:
        print("\nTEST TIDAK DIEVALUASI.")
        print("Gunakan --evaluate-test hanya setelah hyperparameter dan threshold final dikunci.")

    print(f"\nOutput: {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
