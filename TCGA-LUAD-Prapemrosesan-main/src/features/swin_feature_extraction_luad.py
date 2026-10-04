"""Tahap 7: Ekstraksi fitur Swin-Tiny (768-dim) dari patch hasil Macenko. Versi RunPod.

Input : Patches_Macenko/<split>/<label>/<slide>.zip   (patch PNG + manifest.csv)
Output: Features_Swin/h5_files/<slide>.h5   -> dataset 'features' (N,768 fp16), 'coords' (N,2 int32)  [format CLAM]
        Features_Swin/pt_files/<slide>.pt   -> tensor (N,768) float32                                  [untuk ABMIL]
        Features_Swin/features_summary.csv  -> slide, patient_id, label, split, n_patches, dll.

Fitur diekstrak untuk SEMUA slide (train dan test). Pemisahan train/test dilakukan di tahap agregator
memakai kolom split/patient_id pada features_summary.csv.

Uji dulu (3 slide):   python swin_feature_extraction_luad.py --limit 3
Jalankan penuh:       nohup python swin_feature_extraction_luad.py > /workspace/TCGA-LUAD/swin.log 2>&1 &
Resume otomatis: slide yang h5-nya sudah ada dilewati.
"""
import os

# cache bobot di volume agar tidak diunduh ulang setelah Pod di-Stop
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")

import argparse
import gc
import re
import time
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)

# ---------------- KONFIGURASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
DEFAULT_IN_DIR = BASE_DIR / "Patches_Macenko"
DEFAULT_OUT_DIR = BASE_DIR / "Features_Swin"

MODEL_NAME = "swin_tiny_patch4_window7_224"   # Swin-Tiny, input 224x224, fitur 768-dim
PATCH_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = int(os.environ.get("N_WORKERS", 12))
USE_FP16 = True                                # autocast fp16 di GPU
SAVE_PT = True
COORD_RE = re.compile(r"_x(\d+)_y(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


# ---------------- DATASET (baca patch langsung dari zip) ----------------
class ZipPatches(Dataset):
    def __init__(self, zip_path, entries):
        self.zip_path = str(zip_path)
        self.entries = entries
        self.zf = None                     # dibuka lazy di tiap worker (ZipFile tidak aman dibagi antar proses)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path)
        try:
            data = self.zf.read(self.entries[i])
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                raise ValueError("patch tidak valid")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ok = True
        except Exception:
            rgb = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), np.uint8)
            ok = False
        return torch.from_numpy(rgb).permute(2, 0, 1), ok     # uint8 CHW; normalisasi dilakukan di GPU


def read_slide_index(zip_path, stem):
    """Kembalikan (entries, coords, patient_id). Urutan entries = urutan baris manifest."""
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        mname = f"{stem}/manifest.csv"
        man = pd.read_csv(zf.open(mname)) if mname in names else None

    patient_id = ""
    if man is not None and {"patch", "x", "y"} <= set(man.columns) and len(man):
        entries = [f"{stem}/{p}" for p in man["patch"]]
        present = np.array([e in names for e in entries])
        if not present.all():
            print(f"   ! {(~present).sum()} baris manifest tanpa file di zip, dilewati")
        entries = [e for e, ok in zip(entries, present) if ok]
        coords = man.loc[present, ["x", "y"]].to_numpy(np.int64)
        if "patient_id" in man.columns:
            patient_id = str(man["patient_id"].iloc[0])
    else:                                                      # cadangan: ambil koordinat dari nama file
        entries, coords = [], []
        for n in sorted(names):
            m = COORD_RE.search(n)
            if m:
                entries.append(n)
                coords.append((int(m.group(1)), int(m.group(2))))
        coords = np.array(coords, dtype=np.int64).reshape(-1, 2)
    return entries, coords, patient_id


# ---------------- MODEL ----------------
def load_model(device):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)   # num_classes=0 -> fitur pooled 768
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    if tuple(cfg["input_size"][1:]) != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(f"input model {cfg['input_size']} != patch {PATCH_SIZE}")
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    pc = getattr(model, "pretrained_cfg", {}) or {}
    print(f"Model   : {MODEL_NAME} | fitur {model.num_features}-dim | bobot: {pc.get('hf_hub_id', pc.get('url', '?'))}")
    print(f"Normalisasi input: mean={cfg['mean']} std={cfg['std']}")
    return model, mean, std


@torch.inference_mode()
def extract_features(model, mean, std, device, zip_path, entries, batch_size, workers):
    loader = DataLoader(ZipPatches(zip_path, entries), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        prefetch_factor=4 if workers > 0 else None, persistent_workers=False)
    feats, oks = [], []
    for imgs, ok in loader:
        x = imgs.to(device, non_blocking=True).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_FP16):
            f = model(x)
        feats.append(f.to(torch.float16).cpu())
        oks.append(ok)
    return torch.cat(feats), torch.cat(oks)


# ---------------- SIMPAN ----------------
def save_outputs(h5_dir, pt_dir, stem, feats, coords, attrs):
    h5_path, h5_tmp = h5_dir / f"{stem}.h5", h5_dir / f"{stem}.h5.tmp"
    pt_path, pt_tmp = pt_dir / f"{stem}.pt", pt_dir / f"{stem}.pt.tmp"
    if SAVE_PT:
        torch.save(feats.float(), pt_tmp)
        os.replace(pt_tmp, pt_path)
    with h5py.File(h5_tmp, "w") as f:
        f.create_dataset("features", data=feats.numpy())
        f.create_dataset("coords", data=coords.astype(np.int32))
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(h5_tmp, h5_path)                  # h5 terakhir = penanda slide selesai


def save_summary(csv_path, rows):
    tmp = csv_path.with_suffix(".csv.tmp")
    pd.DataFrame(list(rows.values())).to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="proses hanya N slide pertama (untuk uji)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU tidak terdeteksi. Pastikan Pod memakai GPU (cek: nvidia-smi).")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"GPU     : {torch.cuda.get_device_name(0)} | worker {args.workers} | batch {args.batch_size}")

    h5_dir, pt_dir = args.output_dir / "h5_files", args.output_dir / "pt_files"
    summary_csv = args.output_dir / "features_summary.csv"
    for d in (h5_dir, pt_dir):
        d.mkdir(parents=True, exist_ok=True)
    for p in list(h5_dir.glob("*.tmp")) + list(pt_dir.glob("*.tmp")):
        p.unlink()

    zips = sorted(args.input_dir.glob("*/*/*.zip"))
    if not zips:
        raise SystemExit(f"Tidak ada zip di {args.input_dir}. Jalankan macenko_luad.py dulu.")
    if args.limit:
        zips = zips[:args.limit]
    print(f"Zip patch: {len(zips)} | input {args.input_dir} | output {args.output_dir}")

    rows = {}
    if summary_csv.exists():
        try:
            rows = {r["slide"]: r for r in pd.read_csv(summary_csv).to_dict("records")}
        except Exception:
            rows = {}

    model, mean, std = load_model(device)
    print("=" * 70, flush=True)

    ok_n = skip_n = fail_n = 0
    t_all = time.time()
    for n, zpath in enumerate(zips, start=1):
        stem = zpath.stem
        split, label = zpath.relative_to(args.input_dir).parts[:2]
        if (h5_dir / f"{stem}.h5").exists():
            skip_n += 1
            print(f"[{n}/{len(zips)}] SKIP (sudah ada): {stem}")
            continue

        t0 = time.time()
        try:
            entries, coords, patient_id = read_slide_index(zpath, stem)
            if not entries:
                raise ValueError("tidak ada patch di zip")
            feats, ok = extract_features(model, mean, std, device, zpath, entries,
                                         args.batch_size, args.workers)
            good = ok & torch.isfinite(feats.float()).all(dim=1)      # buang patch rusak / fitur NaN
            n_bad = int((~good).sum())
            if n_bad:
                print(f"   ! {n_bad} patch dibuang (gagal decode / fitur tidak valid)")
            if int(good.sum()) == 0:
                raise ValueError("semua patch gagal")
            feats, coords = feats[good], coords[good.numpy()]

            attrs = {"slide": stem, "patient_id": patient_id, "label": label, "split": split,
                     "model": MODEL_NAME, "feature_dim": int(feats.shape[1])}
            save_outputs(h5_dir, pt_dir, stem, feats, coords, attrs)

            dt = time.time() - t0
            rows[stem] = {"slide": stem, "patient_id": patient_id, "label": label, "split": split,
                          "n_patches": int(feats.shape[0]), "n_dropped": n_bad,
                          "feature_dim": int(feats.shape[1]), "time_sec": round(dt, 1),
                          "h5": str(h5_dir / f"{stem}.h5"),
                          "pt": str(pt_dir / f"{stem}.pt") if SAVE_PT else ""}
            save_summary(summary_csv, rows)
            ok_n += 1
            print(f"[{n}/{len(zips)}] OK {stem} ({label}, {split}) | {feats.shape[0]} patch | "
                  f"{feats.shape[0] / dt:.0f} patch/s | {dt:.0f}s", flush=True)
        except KeyboardInterrupt:
            print("\nDihentikan manual. Jalankan ulang untuk melanjutkan.")
            break
        except Exception as e:
            fail_n += 1
            print(f"[{n}/{len(zips)}] GAGAL {stem}: {repr(e)}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"SELESAI dalam {(time.time() - t_all) / 60:.1f} menit | berhasil {ok_n} | dilewati {skip_n} | gagal {fail_n}")
    if rows:
        df = pd.DataFrame(list(rows.values()))
        print(f"Total slide berfitur: {len(df)} | total patch: {int(df['n_patches'].sum())}")
        print(df.groupby(["split", "label"])["slide"].count().to_string())
    print(f"Output: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
