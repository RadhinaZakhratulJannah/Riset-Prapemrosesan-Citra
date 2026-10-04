"""Tahap 4: Tissue detection (HSV Saturation + Otsu) untuk semua WSI. Versi RunPod.

- Membaca .svs langsung dari /workspace (tanpa copy ke disk lokal).
- Paralel per slide (ProcessPool), resume otomatis, tulis atomik (.tmp -> rename).
- Output per slide di BASE_DIR/Tissue_Detection:
    <slide>_tissue_mask.png        mask (tissue=255), dipakai tahap patch extraction
    <slide>_tissue_detection.png   thumbnail | mask | overlay (untuk QC visual)
  + summary_tissue_detection.csv (termasuk scale_x/scale_y untuk konversi koordinat)

Jalankan:  nohup python tissue_detection_luad.py > /workspace/TCGA-LUAD/tissue.log 2>&1 &
"""
import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import openslide
from skimage.morphology import remove_small_objects, remove_small_holes

cv2.setNumThreads(1)          # 1 thread OpenCV per proses; paralelisme dari pool

# ---------------- LOKASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
WSI_INPUT_DIR = BASE_DIR / "WSI"
OUTPUT_DIR = BASE_DIR / "Tissue_Detection"
SUMMARY_CSV = OUTPUT_DIR / "summary_tissue_detection.csv"

# ---------------- PARAMETER ----------------
MASK_DOWNSAMPLE = 32      # thumbnail = level 0 / 32
MIN_OBJECT_SIZE = 500
HOLE_AREA = 5000
CLOSE_KERNEL = 11
OPEN_KERNEL = 5
N_WORKERS = int(os.environ.get("N_WORKERS", min(os.cpu_count() or 4, 8)))


def clean_mask(mask):
    """Kompatibel dengan scikit-image lama dan baru."""
    try:
        mask = remove_small_objects(mask, max_size=MIN_OBJECT_SIZE)
        mask = remove_small_holes(mask, max_size=HOLE_AREA)
    except TypeError:
        mask = remove_small_objects(mask, min_size=MIN_OBJECT_SIZE)
        mask = remove_small_holes(mask, area_threshold=HOLE_AREA)
    return mask


def tissue_detection_otsu(slide_path, mask_downsample=32):
    slide = openslide.OpenSlide(str(slide_path))
    try:
        width, height = slide.dimensions
        thumb_w = max(1, int(width / mask_downsample))
        thumb_h = max(1, int(height / mask_downsample))
        thumbnail = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))

        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
        otsu_threshold, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_u8 = (binary > 0).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open)

        mask = clean_mask(mask_u8 > 0)
        final_mask = mask.astype(np.uint8) * 255

        overlay = thumbnail.copy()
        area = final_mask > 0
        green = np.zeros_like(thumbnail)
        green[:, :, 1] = 255
        overlay[area] = (0.65 * overlay[area] + 0.35 * green[area]).astype(np.uint8)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        return {"thumbnail": thumbnail, "mask": final_mask, "overlay": overlay,
                "otsu_threshold": float(otsu_threshold), "tissue_fraction": float(np.mean(mask)),
                "width": width, "height": height}
    finally:
        slide.close()
        gc.collect()


def combined_path(base):
    return OUTPUT_DIR / f"{base}_tissue_detection.png"


def mask_path(base):
    return OUTPUT_DIR / f"{base}_tissue_mask.png"


def write_png_atomic(path, arr):
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), arr):
        raise IOError(f"gagal menulis {tmp}")
    os.replace(tmp, path)


def load_mask(base):
    """Dipakai tahap patch extraction: kembalikan mask bool (thumbnail resolution)."""
    return cv2.imread(str(mask_path(base)), cv2.IMREAD_GRAYSCALE) > 127


def process_slide(slide_path):
    """Dijalankan di proses worker. Mengembalikan 1 baris summary."""
    slide_path = Path(slide_path)
    base, t0 = slide_path.stem, time.time()
    try:
        r = tissue_detection_otsu(slide_path, MASK_DOWNSAMPLE)
        mask_rgb = cv2.cvtColor(r["mask"], cv2.COLOR_GRAY2RGB)
        combined = np.hstack([r["thumbnail"], mask_rgb, r["overlay"]])
        write_png_atomic(mask_path(base), r["mask"])                       # mask dulu
        write_png_atomic(combined_path(base), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        th, tw = r["thumbnail"].shape[:2]
        return {"slide": base, "status": "BERHASIL",
                "otsu_threshold": r["otsu_threshold"], "tissue_fraction": r["tissue_fraction"],
                "width": r["width"], "height": r["height"],
                "thumb_width": tw, "thumb_height": th,
                "scale_x": r["width"] / tw, "scale_y": r["height"] / th,
                "time_sec": time.time() - t0, "error": ""}
    except Exception as e:
        return {"slide": base, "status": "GAGAL_TISSUE_DETECTION",
                "time_sec": time.time() - t0, "error": repr(e)[:300]}


def outputs_exist(base):
    return mask_path(base).exists() and combined_path(base).exists()


def save_summary(rows):
    new = pd.DataFrame(rows.values())
    tmp = SUMMARY_CSV.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, SUMMARY_CSV)


def main():
    if not WSI_INPUT_DIR.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {WSI_INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUTPUT_DIR.glob("*.tmp.png"):          # sisa tulis yang terputus
        p.unlink()

    svs_files = sorted(WSI_INPUT_DIR.rglob("*.svs"))
    print(f"Ditemukan {len(svs_files)} WSI | worker: {N_WORKERS}")
    if not svs_files:
        raise RuntimeError("Tidak ada file .svs. Jalankan download_wsi_luad.py dulu.")

    rows = {}
    if SUMMARY_CSV.exists():
        try:
            rows = {r["slide"]: r for r in pd.read_csv(SUMMARY_CSV).to_dict("records")}
        except Exception:
            rows = {}

    todo = [p for p in svs_files if not outputs_exist(p.stem)]
    print(f"Sudah selesai: {len(svs_files) - len(todo)} | akan diproses: {len(todo)}")

    ok = fail = 0
    t_all = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        futs = {ex.submit(process_slide, str(p)): p for p in todo}
        for i, f in enumerate(as_completed(futs), start=1):
            row = f.result()
            rows[row["slide"]] = row
            save_summary(rows)
            if row["status"] == "BERHASIL":
                ok += 1
                print(f"[{i}/{len(todo)}] OK  {row['slide']} | tissue {row['tissue_fraction']:.3f} "
                      f"| {row['time_sec']:.1f}s", flush=True)
            else:
                fail += 1
                print(f"[{i}/{len(todo)}] GAGAL {row['slide']}: {row.get('error')}", flush=True)

    print("\n" + "=" * 60)
    print(f"SELESAI dalam {time.time() - t_all:.0f} detik | berhasil {ok} | gagal {fail}")
    print(f"Hasil  : {OUTPUT_DIR}\nSummary: {SUMMARY_CSV}")
    if fail:
        print("Jalankan ulang skrip untuk mencoba slide yang gagal.")


if __name__ == "__main__":
    main()
