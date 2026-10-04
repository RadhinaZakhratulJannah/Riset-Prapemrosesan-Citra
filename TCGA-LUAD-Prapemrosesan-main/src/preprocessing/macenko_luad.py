"""Tahap 6: Macenko color normalization (zip -> zip). Versi RunPod.

Input : Patches/<split>/<label>/<slide>.zip            (dari patch_extraction_luad.py)
Output: Patches_Macenko/<split>/<label>/<slide>.zip    (semua patch ternormalisasi, PNG)
        samples_orig/<slide>.zip                       (SAMPLE_N patch ASLI, lokasi sama dgn hasil normalisasi)

Hemat storage: setelah zip hasil terverifikasi dan sampel asli tersimpan, zip patch asli penuh
DIHAPUS (DELETE_ORIGINAL_AFTER_NORM). Patch asli bisa dibuat ulang dari .svs + koordinat di manifest.

Referensi warna dipilih SEKALI dari slide TRAIN lalu disimpan (reference/reference_params.npz).

Jalankan:  nohup python macenko_luad.py > /workspace/TCGA-LUAD/macenko.log 2>&1 &
Mode stain (patch/slide): env FIT_LEVEL=slide python macenko_luad.py   (jauh lebih cepat)
"""
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
import matplotlib
matplotlib.use("Agg")                   # headless
import matplotlib.pyplot as plt

cv2.setNumThreads(1)

# ---------------- LOKASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
PATCH_DIR = BASE_DIR / "Patches"
OUT_DIR = BASE_DIR / "Patches_Macenko"
SAMPLE_DIR = BASE_DIR / "samples_orig"

REF_DIR = OUT_DIR / "reference"
REF_PARAMS = REF_DIR / "reference_params.npz"
REF_PNG = REF_DIR / "reference.png"
MANIFEST_DIR = OUT_DIR / "manifests"
VALID_DIR = OUT_DIR / "validation"
QC_DIR = OUT_DIR / "qc"
SUMMARY_CSV = OUT_DIR / "macenko_summary.csv"
MANIFEST_ALL = OUT_DIR / "macenko_manifest.csv"

# ---------------- PARAMETER ----------------
IO, ALPHA, BETA = 240, 1, 0.15
MIN_TISSUE_PIXELS = 500

FIT_LEVEL = os.environ.get("FIT_LEVEL", "patch")   # "patch" (per patch, fallback ke slide) atau "slide"
SLIDE_FIT_PATCHES = 30

REFERENCE_PATCH = None
USE_SAVED_REFERENCE = True
REF_SLIDES = 10
REF_PATCHES_PER_SLIDE = 10

OUT_FORMAT = "same"                 # "same", "png", atau "jpg"
JPEG_QUALITY = 95
PNG_COMPRESSION = 3

SAMPLE_N = 100                      # patch asli yang disimpan per slide
DELETE_ORIGINAL_AFTER_NORM = True   # hapus zip asli penuh setelah hasil terverifikasi

VALIDATE_N = 30
N_SHOW = 8
MAX_FAIL_FRAC = 0.01
MAX_MEDIAN_ANGLE = 10.0
MAX_P95_ANGLE = 20.0
MAXC_RATIO_RANGE = (0.85, 1.15)
MAX_BG_SHIFT = 0.05
MAX_CLIP = 0.02

N_WORKERS = int(os.environ.get("N_WORKERS", min(os.cpu_count() or 4, 16)))
MIN_FREE_GB = 30.0
SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg")


# ---------------- MACENKO ----------------
def to_od(pix):
    return -np.log((pix.astype(np.float64) + 1.0) / IO)


def fit_from_od(OD):
    ODhat = OD[~np.any(OD < BETA, axis=1)]
    if ODhat.shape[0] < MIN_TISSUE_PIXELS:
        raise ValueError(f"piksel jaringan terlalu sedikit ({ODhat.shape[0]})")
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    plane = eigvecs[:, 1:3]
    proj = ODhat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [ALPHA, 100 - ALPHA])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    HE = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
    C = np.linalg.pinv(HE) @ OD.T
    maxC = np.percentile(C, 99, axis=1)
    if not (np.all(np.isfinite(HE)) and np.all(np.isfinite(maxC)) and np.all(maxC > 1e-6)):
        raise ValueError("estimasi stain tidak valid")
    return HE, maxC


def macenko_fit(pix):
    return fit_from_od(to_od(pix))


def macenko_normalize(rgb, HE_ref, maxC_ref, HE, maxC):
    h, w, _ = rgb.shape
    OD = to_od(rgb.reshape(-1, 3))
    C = np.linalg.pinv(HE) @ OD.T
    scale = maxC / maxC_ref
    if not np.all(np.isfinite(scale)) or np.any(scale < 1e-6):
        raise ValueError("skala konsentrasi tidak valid")
    out = IO * np.exp(-(HE_ref @ (C / scale[:, None])))
    return np.clip(out, 0, 255).T.reshape(h, w, 3).astype(np.uint8)


# ---------------- BANTU ----------------
def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def decode_rgb(data):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("gagal decode patch")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_rgb(rgb, ext):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ext == ".jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise IOError("gagal encode patch")
    return buf.tobytes()


def out_ext(name):
    if OUT_FORMAT == "same":
        return ".jpg" if Path(name).suffix.lower() in (".jpg", ".jpeg") else ".png"
    return ".jpg" if OUT_FORMAT == "jpg" else ".png"


def image_names(zf):
    return [i.filename for i in zf.infolist() if Path(i.filename).suffix.lower() in IMG_EXT]


def tissue_mask(rgb):
    return ~np.any(to_od(rgb.reshape(-1, 3)) < BETA, axis=1)


def lab_mean(rgb, mask):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab[mask].mean(axis=0)


def angle_deg(u, v):
    c = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def white_frac(rgb):
    return float(np.mean(np.all(rgb >= 220, axis=2)))


def clip_frac(rgb, mask):
    px = rgb.reshape(-1, 3)[mask]
    return float(np.mean(np.any((px == 0) | (px == 255), axis=1))) if len(px) else 0.0


def list_zips():
    return sorted(PATCH_DIR.glob("*/*/*.zip"))


def split_label(zpath):
    rel = zpath.relative_to(PATCH_DIR).parts
    return rel[0], rel[1]


# ---------------- REFERENSI ----------------
def get_reference(zips):
    if REFERENCE_PATCH is not None:
        ref_rgb = load_rgb(REFERENCE_PATCH)
        HE, mC = macenko_fit(ref_rgb.reshape(-1, 3))
        src = f"REFERENCE_PATCH ({REFERENCE_PATCH})"
    elif USE_SAVED_REFERENCE and REF_PARAMS.exists() and REF_PNG.exists():
        z = np.load(REF_PARAMS)
        return z["HE"], z["maxC"], load_rgb(REF_PNG), "referensi tersimpan"
    else:
        if not zips:
            raise SystemExit("Tidak ada zip patch untuk memilih referensi.")
        rng = np.random.default_rng(SEED)
        train = [z for z in zips if split_label(z)[0] == "train"] or zips
        picked = [train[i] for i in rng.permutation(len(train))[:REF_SLIDES]]
        imgs, fits, srcs = [], [], []
        for zp in picked:
            with zipfile.ZipFile(zp) as zf:
                names = image_names(zf)
                for j in rng.permutation(len(names))[:REF_PATCHES_PER_SLIDE]:
                    im = decode_rgb(zf.read(names[j]))
                    try:
                        fit = macenko_fit(im.reshape(-1, 3))
                    except ValueError:
                        continue
                    imgs.append(im); fits.append(fit); srcs.append(f"{zp.stem}/{Path(names[j]).name}")
        if len(fits) < 3:
            raise SystemExit("Terlalu sedikit patch valid untuk memilih referensi.")
        maxCs = np.array([f[1] for f in fits])
        feat = np.hstack([np.array([f[0].ravel() for f in fits]), maxCs / np.median(maxCs, axis=0)])
        best = int(np.argmin(np.linalg.norm(feat - np.median(feat, axis=0), axis=1)))   # medoid
        ref_rgb, (HE, mC) = imgs[best], fits[best]
        src = f"medoid dari {len(fits)} patch train ({srcs[best]})"
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(REF_PARAMS, HE=HE, maxC=mC)
    cv2.imwrite(str(REF_PNG), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))
    return HE, mC, ref_rgb, src


# ---------------- VALIDASI ----------------
def validate_pairs(pairs, ref):
    HE_ref, maxC_ref, ref_lab = ref["HE"], ref["maxC"], ref["lab"]
    rows = []
    for name, o, n in pairs:
        m = tissue_mask(o)
        lab_o, lab_n = lab_mean(o, m), lab_mean(n, m)
        row = {"patch": name,
               "L_before": lab_o[0], "a_before": lab_o[1], "b_before": lab_o[2],
               "L_after": lab_n[0], "a_after": lab_n[1], "b_after": lab_n[2],
               "dE_ref_before": float(np.linalg.norm(lab_o - ref_lab)),
               "dE_ref_after": float(np.linalg.norm(lab_n - ref_lab)),
               "white_before": white_frac(o), "white_after": white_frac(n),
               "clip_after": clip_frac(n, m),
               "shape_ok": bool(n.shape == o.shape and n.dtype == np.uint8),
               "finite_ok": bool(np.all(np.isfinite(n)))}
        try:
            HE_n, maxC_n = macenko_fit(n.reshape(-1, 3))
            row["angle_H"] = angle_deg(HE_n[:, 0], HE_ref[:, 0])
            row["angle_E"] = angle_deg(HE_n[:, 1], HE_ref[:, 1])
            r = maxC_n / maxC_ref
            row["ratio_maxC_H"], row["ratio_maxC_E"] = float(r[0]), float(r[1])
        except ValueError:
            row["angle_H"] = row["angle_E"] = row["ratio_maxC_H"] = row["ratio_maxC_E"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_qc_slide(stem, pairs, ref_rgb):
    if not pairs:
        return
    idx = np.linspace(0, len(pairs) - 1, min(N_SHOW, len(pairs))).astype(int)
    fig, axes = plt.subplots(2, len(idx) + 1, figsize=(2.4 * (len(idx) + 1), 5.2))
    axes[0, 0].imshow(ref_rgb); axes[0, 0].set_title("REFERENSI", fontsize=9)
    for c, j in enumerate(idx, start=1):
        axes[0, c].imshow(pairs[j][1]); axes[0, c].set_title("asli", fontsize=9)
        axes[1, c].imshow(pairs[j][2]); axes[1, c].set_title("Macenko", fontsize=9)
    for a in axes.ravel():
        a.axis("off")
    plt.suptitle(f"{stem} - sebelum vs sesudah Macenko (mode {FIT_LEVEL})")
    plt.tight_layout()
    plt.savefig(QC_DIR / f"{stem}.png", dpi=90)
    plt.close(fig)


# ---------------- NORMALISASI 1 SLIDE ----------------
def normalize_slide(zin_path, tmp_out, stem, ref):
    HE_ref, maxC_ref = ref["HE"], ref["maxC"]
    rng = np.random.default_rng(SEED)
    stats = {"slide": stem, "patches_in": 0, "norm_patch": 0, "norm_slide_fallback": 0, "dropped": 0}
    kept, pairs = {}, []
    cache = {"params": None, "failed": False}

    with zipfile.ZipFile(zin_path) as zin:
        names = image_names(zin)
        stats["patches_in"] = len(names)
        manifest_name = f"{stem}/manifest.csv"
        manifest = pd.read_csv(zin.open(manifest_name)) if manifest_name in zin.namelist() else None
        val_set = {names[i] for i in rng.choice(len(names), min(VALIDATE_N, len(names)), replace=False)}

        def slide_params():
            if cache["failed"]:
                raise ValueError("estimasi stain level slide gagal")
            if cache["params"] is None:
                try:
                    pick = rng.permutation(len(names))[:SLIDE_FIT_PATCHES]
                    pix = np.concatenate([decode_rgb(zin.read(names[j])).reshape(-1, 3) for j in pick])
                    if len(pix) > 2_000_000:
                        pix = pix[rng.choice(len(pix), 2_000_000, replace=False)]
                    cache["params"] = macenko_fit(pix)
                except Exception:
                    cache["failed"] = True
                    raise
            return cache["params"]

        with zipfile.ZipFile(tmp_out, "w", compression=zipfile.ZIP_STORED) as zout:
            for name in names:
                try:
                    rgb = decode_rgb(zin.read(name))
                    if FIT_LEVEL == "slide":
                        HE, mC = slide_params(); mode = "slide"
                    else:
                        try:
                            HE, mC = macenko_fit(rgb.reshape(-1, 3)); mode = "patch"
                        except ValueError:
                            HE, mC = slide_params(); mode = "slide_fallback"
                    out = macenko_normalize(rgb, HE_ref, maxC_ref, HE, mC)
                except Exception:
                    stats["dropped"] += 1
                    continue
                base = Path(name).name
                new_base = Path(base).stem + out_ext(base)
                zout.writestr(f"{stem}/{new_base}", encode_rgb(out, out_ext(base)))
                kept[base] = (new_base, mode)
                stats["norm_patch" if mode in ("patch", "slide") else "norm_slide_fallback"] += 1
                if name in val_set:
                    pairs.append((new_base, rgb, out))

            if manifest is not None:
                manifest = manifest[manifest["patch"].isin(kept)].copy()
                manifest["macenko_mode"] = manifest["patch"].map(lambda p: kept[p][1])
                manifest["patch"] = manifest["patch"].map(lambda p: kept[p][0])
            else:
                manifest = pd.DataFrame({"patch": [v[0] for v in kept.values()],
                                         "macenko_mode": [v[1] for v in kept.values()]})
            zout.writestr(manifest_name, manifest.to_csv(index=False))
    return manifest, pairs, stats, kept


def write_orig_sample(zin_path, stem, kept):
    """Simpan SAMPLE_N patch asli (lokasi sama dengan hasil normalisasi) ke samples_orig/<slide>.zip."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    dest, tmp = SAMPLE_DIR / f"{stem}.zip", SAMPLE_DIR / f"{stem}.zip.tmp"
    rng = np.random.default_rng(SEED)
    with zipfile.ZipFile(zin_path) as zin:
        full = {Path(n).name: n for n in image_names(zin)}
        bases = sorted(b for b in kept if b in full)
        pick = [bases[i] for i in rng.permutation(len(bases))[:SAMPLE_N]]
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zout:
            for b in pick:
                zout.writestr(f"{stem}/{b}", zin.read(full[b]))
    os.replace(tmp, dest)
    return len(pick)


# ---------------- 1 SLIDE (dijalankan di worker) ----------------
def process_zip(job):
    zpath, ref = Path(job[0]), job[1]
    stem = zpath.stem
    split, label = split_label(zpath)
    dest = OUT_DIR / split / label / f"{stem}.zip"
    tmp = dest.with_name(dest.name + ".tmp")
    t0 = time.time()
    try:
        if shutil.disk_usage(OUT_DIR).free < MIN_FREE_GB * 1e9:
            return {"slide": stem, "status": "DISKFULL"}
        dest.parent.mkdir(parents=True, exist_ok=True)

        manifest, pairs, stats, kept = normalize_slide(zpath, tmp, stem, ref)
        n_ok = stats["norm_patch"] + stats["norm_slide_fallback"]
        if n_ok == 0:
            raise ValueError("0 patch berhasil dinormalisasi")

        with zipfile.ZipFile(tmp) as z:                     # verifikasi jumlah patch di zip hasil
            if len(image_names(z)) != n_ok:
                raise IOError("jumlah patch di zip hasil tidak cocok")

        stats["sample_orig"] = write_orig_sample(zpath, stem, kept)
        validate_pairs(pairs, ref).to_csv(VALID_DIR / f"{stem}.csv", index=False)
        save_qc_slide(stem, pairs, ref["rgb"])
        manifest.to_csv(MANIFEST_DIR / f"{stem}.csv", index=False)

        stats["size_mb"] = round(tmp.stat().st_size / 1e6, 1)
        os.replace(tmp, dest)                               # zip hasil ada = slide selesai
        if DELETE_ORIGINAL_AFTER_NORM:
            zpath.unlink(missing_ok=True)
        stats.update(status="OK", label=label, split=split, time_sec=round(time.time() - t0))
        return stats
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"slide": stem, "status": "GAGAL", "error": repr(e)[:300]}
    finally:
        gc.collect()


def save_summary(stats):
    new = pd.DataFrame([stats])
    if SUMMARY_CSV.exists():
        try:
            new = pd.concat([pd.read_csv(SUMMARY_CSV), new], ignore_index=True)
        except Exception:
            pass
    new = new.drop_duplicates("slide", keep="last")
    tmp = SUMMARY_CSV.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, SUMMARY_CSV)


def merge_manifests():
    frames = [pd.read_csv(f) for f in sorted(MANIFEST_DIR.glob("*.csv")) if f.stat().st_size > 0]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(MANIFEST_ALL, index=False)
        print(f"Manifest gabungan: {MANIFEST_ALL}")


# ---------------- LAPORAN VALIDASI ----------------
def final_report(ref):
    files = sorted(VALID_DIR.glob("*.csv"))
    if not files or not SUMMARY_CSV.exists():
        print("Belum ada data validasi.")
        return
    rep = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summ = pd.read_csv(SUMMARY_CSV)
    rep.to_csv(OUT_DIR / "macenko_report_all.csv", index=False)

    n_in, n_drop = int(summ["patches_in"].sum()), int(summ["dropped"].sum())
    fail_frac = n_drop / max(n_in, 1)
    fb_frac = summ["norm_slide_fallback"].sum() / max(n_in, 1)
    ang = np.concatenate([rep["angle_H"].dropna().values, rep["angle_E"].dropna().values])
    ratio = np.concatenate([rep["ratio_maxC_H"].dropna().values, rep["ratio_maxC_E"].dropna().values])
    std_b = rep[["L_before", "a_before", "b_before"]].std().sum()
    std_a = rep[["L_after", "a_after", "b_after"]].std().sum()
    bg_shift = float((rep["white_after"] - rep["white_before"]).abs().mean())

    checks = [
        ("1. Hampir semua patch berhasil dinormalisasi", fail_frac <= MAX_FAIL_FRAC,
         f"{n_in - n_drop}/{n_in} berhasil, dibuang {fail_frac * 100:.2f}% (batas {MAX_FAIL_FRAC * 100:.0f}%); "
         f"fallback ke stain slide: {fb_frac * 100:.2f}%"),
        ("2. Ukuran, tipe uint8, dan nilai valid (tanpa NaN)", bool(rep["shape_ok"].all() and rep["finite_ok"].all()),
         f"{len(rep)} patch divalidasi dari {len(files)} slide"),
        ("3. Arah vektor stain hasil ~ referensi",
         bool(len(ang) and np.median(ang) <= MAX_MEDIAN_ANGLE and np.percentile(ang, 95) <= MAX_P95_ANGLE),
         f"median {np.median(ang):.1f} deg, p95 {np.percentile(ang, 95):.1f} deg "
         f"(batas {MAX_MEDIAN_ANGLE:.0f}/{MAX_P95_ANGLE:.0f})" if len(ang) else "tidak bisa dihitung"),
        ("4. Intensitas stain hasil ~ referensi",
         bool(len(ratio) and MAXC_RATIO_RANGE[0] <= np.median(ratio) <= MAXC_RATIO_RANGE[1]),
         f"rasio konsentrasi maks median {np.median(ratio):.2f} (harus {MAXC_RATIO_RANGE[0]}-{MAXC_RATIO_RANGE[1]})"
         if len(ratio) else "tidak bisa dihitung"),
        ("5. Warna jaringan lebih dekat ke referensi", bool(rep["dE_ref_after"].median() < rep["dE_ref_before"].median()),
         f"jarak LAB ke referensi median {rep['dE_ref_before'].median():.1f} -> {rep['dE_ref_after'].median():.1f}"),
        ("6. Variasi warna antar patch/slide tidak memburuk", bool(std_a <= std_b * 1.10),
         f"total std LAB {std_b:.2f} -> {std_a:.2f}"),
        ("7. Latar putih tidak berubah", bool(bg_shift <= MAX_BG_SHIFT),
         f"rata-rata pergeseran porsi latar {bg_shift:.3f} (batas {MAX_BG_SHIFT})"),
        ("8. Tidak ada clipping berlebihan", bool(rep["clip_after"].mean() <= MAX_CLIP),
         f"porsi piksel jaringan 0/255: {rep['clip_after'].mean() * 100:.2f}% (batas {MAX_CLIP * 100:.0f}%)"),
    ]
    print("\n" + "=" * 78)
    print("HASIL VALIDASI MACENKO (SELURUH SLIDE)")
    print("=" * 78)
    for name, ok, info in checks:
        print(f"[{'LULUS' if ok else 'PERIKSA'}] {name}\n          {info}")
    n_ok = sum(ok for _, ok, _ in checks)
    print("-" * 78)
    print(f"Kesimpulan: {n_ok}/{len(checks)} pemeriksaan lulus"
          + (" -> hasil normalisasi dapat dipakai." if n_ok == len(checks)
             else " -> lihat butir PERIKSA dan gambar QC sebelum melanjutkan."))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(rep["a_before"], rep["b_before"], s=12, label="sebelum", alpha=0.5)
    ax[0].scatter(rep["a_after"], rep["b_after"], s=12, label="sesudah", alpha=0.5)
    ax[0].scatter([ref["lab"][1]], [ref["lab"][2]], marker="*", s=220, c="red", label="referensi")
    ax[0].set_xlabel("a*"); ax[0].set_ylabel("b*")
    ax[0].set_title("Warna jaringan per patch (LAB): harus mengumpul"); ax[0].legend()
    ax[1].hist(rep["dE_ref_before"], bins=25, alpha=0.7, label="sebelum")
    ax[1].hist(rep["dE_ref_after"], bins=25, alpha=0.7, label="sesudah")
    ax[1].set_xlabel("jarak warna ke referensi (LAB)"); ax[1].set_title("Harus bergeser ke kiri"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc_sebaran_warna.png", dpi=110)
    plt.close(fig)
    print(f"Laporan: {OUT_DIR / 'macenko_report_all.csv'} | grafik: {OUT_DIR / 'qc_sebaran_warna.png'}")


# ---------------- MAIN ----------------
def main():
    for d in (OUT_DIR, REF_DIR, MANIFEST_DIR, VALID_DIR, QC_DIR, SAMPLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for p in list(OUT_DIR.rglob("*.zip.tmp")) + list(SAMPLE_DIR.glob("*.zip.tmp")):
        p.unlink()

    zips = list_zips()
    print(f"Zip patch asli tersisa: {len(zips)}")
    HE_ref, maxC_ref, ref_rgb, src = get_reference(zips)
    ref = {"HE": HE_ref, "maxC": maxC_ref, "rgb": ref_rgb, "lab": lab_mean(ref_rgb, tissue_mask(ref_rgb))}
    print(f"Referensi : {src}")

    todo = []
    for zp in zips:
        split, label = split_label(zp)
        dest = OUT_DIR / split / label / f"{zp.stem}.zip"
        if dest.exists():                                   # selesai; bereskan zip asli yang tertinggal
            if DELETE_ORIGINAL_AFTER_NORM:
                zp.unlink(missing_ok=True)
        else:
            todo.append(zp)
    print("=" * 70)
    print(f"MACENKO | mode {FIT_LEVEL} | worker {N_WORKERS} | akan diproses: {len(todo)} | "
          f"hapus asli setelah selesai: {DELETE_ORIGINAL_AFTER_NORM}")
    print(f"Ruang kosong: {shutil.disk_usage(OUT_DIR).free / 1e9:.0f} GB")
    print("=" * 70, flush=True)

    ok = fail = 0
    stopped = False
    ex = ProcessPoolExecutor(N_WORKERS)
    try:
        futs = [ex.submit(process_zip, (str(zp), ref)) for zp in todo]
        for n, f in enumerate(as_completed(futs), start=1):
            s = f.result()
            if s["status"] == "DISKFULL":
                print("\nBERHENTI: ruang volume hampir habis. Kosongkan/perbesar volume, lalu jalankan ulang.")
                stopped = True
                break
            if s["status"] == "OK":
                ok += 1
                save_summary(s)
                print(f"[{n}/{len(todo)}] OK {s['slide']} | {s['norm_patch'] + s['norm_slide_fallback']}/"
                      f"{s['patches_in']} patch ({s['size_mb']:.0f} MB) | fallback {s['norm_slide_fallback']} | "
                      f"dibuang {s['dropped']} | {s['time_sec']}s", flush=True)
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
    print(f"SELESAI | berhasil {ok} | gagal {fail} | Hasil: {OUT_DIR}")
    print("=" * 70)
    final_report(ref)


if __name__ == "__main__":
    main()
