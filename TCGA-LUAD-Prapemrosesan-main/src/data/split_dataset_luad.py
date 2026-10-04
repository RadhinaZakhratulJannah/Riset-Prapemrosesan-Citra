"""Tahap 3: split 80:20 per PASIEN, stratified per label. Versi RunPod (/workspace)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- KONFIGURASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
WSI_DIR = BASE_DIR / "WSI"
LABELS_CSV = WSI_DIR / "labels.csv"          # dibuat oleh download_wsi_luad.py
SPLIT_CSV = WSI_DIR / "split.csv"

TRAIN_RATIO = 0.80
SEED = 42

# "csv"  : hanya tulis split.csv (disarankan, file .svs tidak dipindah)
# "move" : pindahkan .svs ke WSI/train/<label>/ dan WSI/test/<label>/
# "undo" : kembalikan semua .svs ke WSI/
MODE = "csv"


def make_split(df):
    """Split per pasien & stratified per label -> tidak ada kebocoran data pasien."""
    rng = np.random.default_rng(SEED)
    pat = df.groupby("patient_id")["label"].first().reset_index()

    train_ids = []
    for lab, g in pat.groupby("label"):
        ids = np.sort(g["patient_id"].to_numpy())     # urut dulu -> hasil deterministik
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * TRAIN_RATIO))
        if n >= 2:
            n_train = min(max(n_train, 1), n - 1)
        train_ids.extend(ids[:n_train])

    out = df.copy()
    out["split"] = np.where(out["patient_id"].isin(set(train_ids)), "train", "test")
    return out


def check_split(out):
    tr = set(out.loc[out["split"] == "train", "patient_id"])
    te = set(out.loc[out["split"] == "test", "patient_id"])
    assert not (tr & te), "Ada pasien di train dan test sekaligus!"
    assert out["file_name"].is_unique, "Ada file_name duplikat"


def summarize(out):
    print("=" * 56)
    print("RINGKASAN SPLIT")
    print("=" * 56)
    tab = pd.crosstab(out["label"], out["split"]).reindex(columns=["train", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab.loc["TOTAL"] = tab.sum()
    print(tab.to_string())
    for s in ("train", "test"):
        d = out[out["split"] == s]
        if len(d):
            print(f"  {s:5s}: {len(d)} slide | {(d['label'] == 'Mutant').mean() * 100:.1f}% Mutant")


def move_files(out):
    moved = 0
    for _, r in out.iterrows():
        src = WSI_DIR / r["file_name"]
        dst = WSI_DIR / r["split"] / r["label"] / r["file_name"]
        if dst.exists():
            continue
        if not src.exists():
            print(f"  ! tidak ditemukan: {src.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        moved += 1
    print(f"\n{moved} file dipindahkan ke {WSI_DIR}/train dan {WSI_DIR}/test")


def undo_move():
    moved = 0
    for s in ("train", "test"):
        for p in (WSI_DIR / s).rglob("*.svs"):
            os.replace(p, WSI_DIR / p.name)
            moved += 1
    for s in ("train", "test"):
        shutil.rmtree(WSI_DIR / s, ignore_errors=True)
    print(f"{moved} file dikembalikan ke {WSI_DIR}")


def main():
    if MODE == "undo":
        undo_move()
        return
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"{LABELS_CSV} tidak ada. Jalankan download_wsi_luad.py dulu.")

    df = pd.read_csv(LABELS_CSV)
    if df.empty:
        raise SystemExit("labels.csv kosong, belum ada slide yang selesai diunduh.")

    def exists(name):
        return (WSI_DIR / name).exists() or any(WSI_DIR.glob(f"*/*/{name}"))
    ada = df["file_name"].map(exists)
    if (~ada).any():
        print(f"PERINGATAN: {(~ada).sum()} file di labels.csv tidak ada di disk, dilewati.")
    df = df[ada].copy()

    out = make_split(df)
    check_split(out)
    summarize(out)

    out = out.sort_values(["split", "label", "file_name"])
    tmp = SPLIT_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, SPLIT_CSV)
    print(f"\nSplit disimpan: {SPLIT_CSV}")

    if MODE == "move":
        move_files(out)
    else:
        print("MODE = 'csv': file .svs tidak dipindah. Baca split.csv di tahap berikutnya.")


if __name__ == "__main__":
    main()
