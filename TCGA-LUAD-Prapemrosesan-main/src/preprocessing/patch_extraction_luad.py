"""Tahap 5: Patch extraction 224x224 (level 0, tanpa resize). Versi RunPod.

Filter: tissue < MIN_TISSUE dibuang, blur (variance of Laplacian) < BLUR_THRESHOLD dibuang.
Alur per slide: WSI (/workspace) -> <slide>.zip.tmp -> rename -> manifest csv (penanda selesai).
Hasil: Patches/<split>/<label>/<slide>.zip (isi: patch + manifest.csv)

Kalibrasi dulu (tidak menyimpan patch):   python patch_extraction_luad.py --calibrate
Jalankan penuh:  nohup python patch_extraction_luad.py > /workspace/TCGA-LUAD/patch.log 2>&1 &
"""
import argparse
import gc
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import openslide

cv2.setNumThreads(1)

# ---------------- LOKASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
WSI_DIR = BASE_DIR / "WSI"
TISSUE_DIR = BASE_DIR / "Tissue_Detection"
SUMMARY_CSV = TISSUE_DIR / "summary_tissue_detection.csv"
SPLIT_CSV = WSI_DIR / "split.csv"
LABELS_CSV = WSI_DIR / "labels.csv"

PATCH_DIR = BASE_DIR / "Patches"
MANIFEST_DIR = PATCH_DIR / "manifests"
PATCH_SUMMARY_CSV = PATCH_DIR / "patch_summary.csv"
PATCH_MANIFEST_CSV = PATCH_DIR / "patch_manifest.csv"

# ---------------- PARAMETER ----------------
PATCH_SIZE = 224
STRIDE = 224
LEVEL = 0
MIN_TISSUE = 0.70
PRE_MASK_MIN = 0.50
BLUR_THRESHOLD = 100.0          # kalibrasi dulu dengan --calibrate

PATCH_FORMAT = "png"            # "png" (lossless) atau "jpg"
JPEG_QUALITY = 95
PNG_COMPRESSION = 3

N_WORKERS = int(os.environ.get("N_WORKERS", min(os.cpu_count() or 4, 16)))
MIN_FREE_GB = 30.0              # berhenti rapi jika sisa ruang volume di bawah ini

CALIBRATE_SLIDES = 3
CALIBRATE_SAMPLES = 300
SEED = 42


# ---------------- MASK & GRID ----------------
def load_mask(stem, info):
    tw, th = int(info["thumb_width"]), int(info["thumb_height"])
    mp = TISSUE_DIR / f"{stem}_tissue_mask.png"
    if mp.exists():
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
    else:
        img = cv2.imread(str(TISSUE_DIR / f"{stem}_tissue_detection.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"hasil tissue detection tidak ada: {stem}")
        mask = img[:, tw:2 * tw, 0] > 127
    if mask.shape != (th, tw):
        raise ValueError(f"ukuran mask {mask.shape} != ({th}, {tw})")
    return mask


def candidate_grid(mask, width, height, sx, sy):
    xs = np.arange(0, width - PATCH_SIZE + 1, STRIDE)
    ys = np.arange(0, height - PATCH_SIZE + 1, STRIDE)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    n_grid = len(gx)
    mh, mw = mask.shape
    x0 = np.clip(np.floor(gx / sx).astype(int), 0, mw)
    x1 = np.clip(np.ceil((gx + PATCH_SIZE) / sx).astype(int), 0, mw)
    y0 = np.clip(np.floor(gy / sy).astype(int), 0, mh)
    y1 = np.clip(np.ceil((gy + PATCH_SIZE) / sy).astype(int), 0, mh)
    integ = cv2.integral(mask.astype(np.uint8))
    s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    cov = s / np.maximum((x1 - x0) * (y1 - y0), 1)
    keep = cov >= PRE_MASK_MIN
    return gx[keep], gy[keep], cov[keep], n_grid


# ---------------- FILTER & ENCODE ----------------
def read_patch(slide, x, y):
    return np.array(slide.read_region((int(x), int(y)), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB"))


def tissue_fraction(rgb, sat_threshold):
    sat = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1], (5, 5), 0)
    return float(np.mean(sat > sat_threshold))


def blur_score(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def encode_patch(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if PATCH_FORMAT == "jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise IOError("gagal meng-encode patch")
    return buf.tobytes()


def sat_threshold_of(info):
    t = info.get("otsu_threshold", np.nan)
    return 20.0 if pd.isna(t) else float(t)


# ---------------- DAFTAR SLIDE ----------------
def load_slide_table():
    if SPLIT_CSV.exists():
        meta = pd.read_csv(SPLIT_CSV)
    elif LABELS_CSV.exists():
        meta = pd.read_csv(LABELS_CSV)
        meta["split"] = "all"
        print("PERINGATAN: split.csv tidak ada, memakai labels.csv (folder 'all').")
    else:
        raise FileNotFoundError("split.csv / labels.csv tidak ditemukan di folder WSI.")
    meta["slide"] = meta["file_name"].map(lambda n: Path(n).stem)

    summ = pd.read_csv(SUMMARY_CSV)
    summ = summ[summ["status"] == "BERHASIL"].set_index("slide")
    wsi_index = {p.stem: p for p in WSI_DIR.rglob("*.svs")}

    rows = []
    for _, r in meta.iterrows():
        s = r["slide"]
        if s in summ.index and s in wsi_index:
            rows.append((s, str(wsi_index[s]), summ.loc[s].to_dict(), r.to_dict()))
        else:
            print(f"  lewati {s}: belum ada di summary tissue detection atau WSI tidak ditemukan")
    return rows


# ---------------- KALIBRASI ----------------
def calibrate(rows):
    rng = np.random.default_rng(SEED)
    fracs, blurs, imgs = [], [], []
    for stem, wsi_path, info, _ in rows[:CALIBRATE_SLIDES]:
        mask = load_mask(stem, info)
        gx, gy, _, _ = candidate_grid(mask, int(info["width"]), int(info["height"]),
                                      float(info["scale_x"]), float(info["scale_y"]))
        idx = rng.permutation(len(gx))[:CALIBRATE_SAMPLES]
        T = sat_threshold_of(info)
        slide = openslide.OpenSlide(wsi_path)
        try:
            for i in idx:
                rgb = read_patch(slide, gx[i], gy[i])
                fracs.append(tissue_fraction(rgb, T))
                blurs.append(blur_score(rgb))
                imgs.append(rgb)
        finally:
            slide.close()
        print(f"  kalibrasi {stem}: {len(idx)} patch")

    fracs, blurs = np.array(fracs), np.array(blurs)
    ok = fracs >= MIN_TISSUE
    print(f"\nSampel patch: {len(fracs)}")
    print("Tissue fraction persentil 5/25/50/75:", np.percentile(fracs, [5, 25, 50, 75]).round(2))
    print(f"Lolos tissue >= {MIN_TISSUE:.0%}: {ok.mean() * 100:.1f}%")
    if ok.any():
        print("Blur score (lolos tissue) persentil 5/10/25/50:", np.percentile(blurs[ok], [5, 10, 25, 50]).round(1))
        print(f"BLUR_THRESHOLD = {BLUR_THRESHOLD} -> membuang {(blurs[ok] < BLUR_THRESHOLD).mean() * 100:.1f}% "
              "dari patch yang lolos tissue")
        order = np.argsort(np.where(ok, blurs, np.inf))[:12]
        tiles = []
        for i in order:
            t = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR).copy()
            cv2.putText(t, f"blur={blurs[i]:.0f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        out = PATCH_DIR / "calibrasi_blur_terendah.png"
        cv2.imwrite(str(out), grid)
        print(f"Gambar 12 patch blur terendah: {out}  (unduh/lihat di JupyterLab)")
        print("Naikkan BLUR_THRESHOLD jika patch yang masih tajam ikut terbuang.")


# ---------------- EKSTRAKSI 1 SLIDE (dijalankan di worker) ----------------
def extract_slide(job):
    stem, wsi_path, info, meta = job
    t0 = time.time()
    split, label = str(meta.get("split", "all")), str(meta.get("label", "unknown"))
    dest_dir = PATCH_DIR / split / label
    dest, tmp = dest_dir / f"{stem}.zip", dest_dir / f"{stem}.zip.tmp"
    ext = "jpg" if PATCH_FORMAT == "jpg" else "png"
    try:
        if shutil.disk_usage(PATCH_DIR).free < MIN_FREE_GB * 1e9:
            return {"slide": stem, "status": "DISKFULL"}
        dest_dir.mkdir(parents=True, exist_ok=True)
        width, height = int(info["width"]), int(info["height"])
        T = sat_threshold_of(info)
        mask = load_mask(stem, info)
        gx, gy, cov, n_grid = candidate_grid(mask, width, height,
                                             float(info["scale_x"]), float(info["scale_y"]))
        manifest, rej_t, rej_b = [], 0, 0
        slide = openslide.OpenSlide(wsi_path)
        try:
            if tuple(slide.dimensions) != (width, height):
                raise ValueError(f"dimensi WSI {slide.dimensions} != summary {(width, height)}")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                for i in range(len(gx)):
                    x, y = int(gx[i]), int(gy[i])
                    rgb = read_patch(slide, x, y)
                    tf = tissue_fraction(rgb, T)
                    if tf < MIN_TISSUE:
                        rej_t += 1
                        continue
                    bs = blur_score(rgb)
                    if bs < BLUR_THRESHOLD:
                        rej_b += 1
                        continue
                    fname = f"{stem}_x{x}_y{y}.{ext}"
                    zf.writestr(f"{stem}/{fname}", encode_patch(rgb))
                    manifest.append({"patch": fname, "slide": stem,
                                     "patient_id": meta.get("patient_id", ""), "label": label, "split": split,
                                     "x": x, "y": y, "size": PATCH_SIZE, "level": LEVEL,
                                     "mask_coverage": round(float(cov[i]), 3),
                                     "tissue_fraction": round(tf, 3), "blur_score": round(bs, 1)})
                zf.writestr(f"{stem}/manifest.csv", pd.DataFrame(manifest).to_csv(index=False))
        finally:
            slide.close()
        size = tmp.stat().st_size
        os.replace(tmp, dest)
        mtmp = MANIFEST_DIR / f"{stem}.csv.tmp"
        pd.DataFrame(manifest).to_csv(mtmp, index=False)
        os.replace(mtmp, MANIFEST_DIR / f"{stem}.csv")          # penanda selesai (terakhir)
        return {"slide": stem, "status": "OK", "label": label, "split": split, "grid_cells": n_grid,
                "after_mask": len(gx), "rejected_tissue": rej_t, "rejected_blur": rej_b,
                "saved": len(manifest), "size_mb": round(size / 1e6, 1), "time_sec": round(time.time() - t0)}
    except Exception as e:
        Path(tmp).unlink(missing_ok=True)
        return {"slide": stem, "status": "GAGAL", "error": repr(e)[:300]}
    finally:
        gc.collect()


# ---------------- RINGKASAN ----------------
def save_patch_summary(stats):
    new = pd.DataFrame([stats])
    if PATCH_SUMMARY_CSV.exists():
        try:
            new = pd.concat([pd.read_csv(PATCH_SUMMARY_CSV), new], ignore_index=True)
        except Exception:
            pass
    new = new.drop_duplicates("slide", keep="last")
    tmp = PATCH_SUMMARY_CSV.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, PATCH_SUMMARY_CSV)


def merge_manifests():
    frames = [pd.read_csv(f) for f in sorted(MANIFEST_DIR.glob("*.csv")) if f.stat().st_size > 0]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(PATCH_MANIFEST_CSV, index=False)
        print(f"Manifest gabungan: {PATCH_MANIFEST_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for p in PATCH_DIR.rglob("*.zip.tmp"):
        p.unlink()

    rows = load_slide_table()
    print(f"Slide siap diproses: {len(rows)}")
    if not rows:
        raise SystemExit("Tidak ada slide. Pastikan tissue detection dan split sudah dijalankan.")
    if args.calibrate:
        calibrate(rows)
        return

    todo = [r for r in rows if not (MANIFEST_DIR / f"{r[0]}.csv").exists()]
    print("=" * 70)
    print(f"PATCH {PATCH_SIZE}x{PATCH_SIZE} | tissue >= {MIN_TISSUE:.0%} | blur >= {BLUR_THRESHOLD} | "
          f"{PATCH_FORMAT} | worker {N_WORKERS}")
    print(f"Sudah selesai: {len(rows) - len(todo)} | akan diproses: {len(todo)} | "
          f"ruang kosong {shutil.disk_usage(PATCH_DIR).free / 1e9:.0f} GB")
    print("=" * 70, flush=True)

    ok = fail = 0
    stopped = False
    ex = ProcessPoolExecutor(N_WORKERS)
    try:
        futs = [ex.submit(extract_slide, j) for j in todo]
        for n, f in enumerate(as_completed(futs), start=1):
            s = f.result()
            if s["status"] == "DISKFULL":
                print("\nBERHENTI: ruang volume hampir habis. Kosongkan/perbesar volume, lalu jalankan ulang.")
                stopped = True
                break
            if s["status"] == "OK":
                ok += 1
                save_patch_summary(s)
                print(f"[{n}/{len(todo)}] OK {s['slide']} | {s['saved']} patch ({s['size_mb']:.0f} MB) | "
                      f"dibuang tissue={s['rejected_tissue']} blur={s['rejected_blur']} | {s['time_sec']}s", flush=True)
            else:
                fail += 1
                print(f"[{n}/{len(todo)}] GAGAL {s['slide']}: {s.get('error')}", flush=True)
    except KeyboardInterrupt:
        print("\nDihentikan manual. Jalankan ulang untuk melanjutkan.")
        stopped = True
    finally:
        ex.shutdown(wait=not stopped, cancel_futures=True)
        merge_manifests()

    print("\n" + "=" * 70)
    print(f"SELESAI | berhasil {ok} | gagal {fail} | Patch: {PATCH_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
