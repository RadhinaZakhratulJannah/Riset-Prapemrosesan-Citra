"""Tahap 2: unduh 50 Mutant + 150 Wildtype WSI 40x dari GDC langsung ke Network volume RunPod.
Resume-aman: jalankan ulang kapan saja, file yang selesai dilewati, .part dilanjutkan (HTTP Range).

Jalankan di tmux/nohup:  nohup python download_wsi_luad.py > /workspace/TCGA-LUAD/download.log 2>&1 &
"""
import os
import re
import time
import struct
import hashlib
import threading
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ---------------- KONFIGURASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
LABELED_CSV = BASE_DIR / "gdc_check_output" / "labeled_all_dx.csv"
WSI_DIR = BASE_DIR / "WSI"          # tujuan akhir (Network volume)
PART_DIR = BASE_DIR / "parts"       # .part di volume yang sama -> rename instan, resume tahan Stop Pod
GDC_DATA = "https://api.gdc.cancer.gov/data/"

N_MUTANT = 50
N_WILDTYPE = 150
TARGET_MAG = 40
ONE_SLIDE_PER_PATIENT = True
MIN_SIZE_MB = 0
MAX_SIZE_MB = None

MAX_WORKERS = 4                     # unduhan paralel (aman di RunPod; turunkan ke 2 jika sering 429/timeout)
MAX_RETRIES = 5
BACKOFF_BASE = 5
RETRY_FAILED_ON_RESUME = True
DISK_MARGIN = 20e9                  # sisakan 20 GB di volume
DRY_RUN = False                     # True = hanya tampilkan rencana & total ukuran

PROGRESS_CSV = WSI_DIR / "download_progress.csv"
PLAN_CSV = WSI_DIR / "download_plan.csv"
LABELS_CSV = WSI_DIR / "labels.csv"

_lock = threading.Lock()
plan = None


class NonRetryable(Exception):
    pass


class DiskFull(Exception):
    pass


def log(msg):
    with _lock:
        print(msg, flush=True)


# ---------------- PARSER SVS ----------------
class LocalReader:
    def __init__(self, path):
        self.path = path

    def read(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)


def first_ifd_description(rd):
    h = rd.read(0, 16)
    if h[:2] == b"II":
        e = "<"
    elif h[:2] == b"MM":
        e = ">"
    else:
        raise ValueError(f"bukan TIFF (header={h[:8].hex()})")
    magic = struct.unpack(e + "H", h[2:4])[0]
    if magic == 42:
        off = struct.unpack(e + "I", h[4:8])[0]
        n = struct.unpack(e + "H", rd.read(off, 2))[0]
        esz, cnt_len, val_len, ptr_fmt, hdr = 12, 4, 4, "I", 2
    elif magic == 43:
        off = struct.unpack(e + "Q", h[8:16])[0]
        n = struct.unpack(e + "Q", rd.read(off, 8))[0]
        esz, cnt_len, val_len, ptr_fmt, hdr = 20, 8, 8, "Q", 8
    else:
        raise ValueError(f"magic TIFF tidak dikenal: {magic}")
    if not 0 < n < 1000:
        raise ValueError(f"jumlah entri IFD tidak masuk akal: {n}")
    raw = rd.read(off + hdr, n * esz)
    for i in range(n):
        ent = raw[i * esz:(i + 1) * esz]
        if struct.unpack(e + "H", ent[0:2])[0] != 270:
            continue
        count = struct.unpack(e + ("I" if cnt_len == 4 else "Q"), ent[4:4 + cnt_len])[0]
        val = ent[4 + cnt_len:4 + cnt_len + val_len]
        data = val[:count] if count <= val_len else rd.read(
            struct.unpack(e + ptr_fmt, val)[0], min(count, 64 * 1024))
        return data.split(b"\x00")[0].decode("latin-1", errors="replace")
    raise ValueError("tag ImageDescription tidak ada")


def parse_mag(desc):
    m = re.search(r"AppMag\s*=\s*(\d+)", desc)
    if m:
        return int(m.group(1))
    m = re.search(r"MPP\s*=\s*([\d.]+)", desc)
    if m:
        return 40 if float(m.group(1)) < 0.35 else 20
    return None


def verify_svs(path):
    try:
        desc = first_ifd_description(LocalReader(path))
    except Exception as e:
        return False, f"struktur SVS rusak: {e}"
    mag = parse_mag(desc)
    if mag != TARGET_MAG:
        return False, f"magnifikasi {mag} != {TARGET_MAG}"
    return True, ""


def file_md5(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------- RENCANA SELEKSI ----------------
def build_plan():
    df = pd.read_csv(LABELED_CSV)
    mag = pd.to_numeric(df["magnification"], errors="coerce")
    primary = df["sample_type"].isna() | df["sample_type"].eq("Primary Tumor")
    df = df[mag.eq(TARGET_MAG) & primary].copy()
    df["size_mb"] = df["file_size"] / 1e6
    df = df[df["size_mb"] >= MIN_SIZE_MB]
    if MAX_SIZE_MB:
        df = df[df["size_mb"] <= MAX_SIZE_MB]
    df["label_out"] = df["label"].map({"Mutant": "Mutant"}).fillna("Wildtype")
    df["label_source"] = df["label"].map({"Mutant": "excel_egfr_mutant"}).fillna("not_in_excel")

    parts = []
    for lab, quota in (("Mutant", N_MUTANT), ("Wildtype", N_WILDTYPE)):
        d = df[df["label_out"] == lab].sort_values(["file_size", "file_name"], ascending=[False, True])
        if ONE_SLIDE_PER_PATIENT:
            d = d.drop_duplicates("patient_id")
        d = d.copy()
        d["rank"] = range(1, len(d) + 1)
        d["selected"] = d["rank"] <= quota
        if len(d) < quota:
            print(f"PERINGATAN: kandidat {lab} hanya {len(d)} (< target {quota}).")
        parts.append(d)
    cols = ["file_id", "file_name", "file_size", "size_mb", "md5sum", "patient_id",
            "label_out", "label_source", "rank", "selected"]
    out = pd.concat(parts)[cols].set_index("file_id")
    out["status"] = "pending"
    out["attempts"] = 0
    out["error"] = ""
    out["finished_at"] = ""
    return out


def load_progress(plan):
    if PROGRESS_CSV.exists():
        old = pd.read_csv(PROGRESS_CSV).set_index("file_id")
        common = plan.index.intersection(old.index)
        plan.loc[common, "status"] = old.loc[common, "status"].astype(object)
        plan.loc[common, "attempts"] = old.loc[common, "attempts"].fillna(0).astype(int)
        plan.loc[common, "error"] = old.loc[common, "error"].fillna("").astype(object)
        plan.loc[common, "finished_at"] = old.loc[common, "finished_at"].fillna("").astype(object)
    if RETRY_FAILED_ON_RESUME:
        plan.loc[plan["status"] == "failed", "status"] = "pending"
    return plan


def save_progress():
    tmp = PROGRESS_CSV.with_suffix(".csv.tmp")
    plan.to_csv(tmp)
    os.replace(tmp, PROGRESS_CSV)


def set_state(fid, status, attempts, error=""):
    with _lock:
        plan.at[fid, "status"] = status
        plan.at[fid, "attempts"] = attempts
        plan.at[fid, "error"] = error[:300]
        plan.at[fid, "finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S") if status == "done" else ""
        save_progress()


def write_labels():
    d = plan[plan["status"] == "done"].reset_index()
    d = d[["file_name", "patient_id", "label_out", "label_source", "size_mb"]]
    d = d.rename(columns={"label_out": "label"}).sort_values(["label", "file_name"])
    d.to_csv(LABELS_CSV, index=False)


# ---------------- DOWNLOAD ----------------
def make_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    ad = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


def stream_to_part(session, fid, part, have):
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(GDC_DATA + fid, headers=headers, stream=True, timeout=(30, 120)) as r:
        if r.status_code == 206:
            mode = "ab"
        else:
            r.raise_for_status()
            mode = "wb"                     # server mengabaikan Range -> mulai dari awal
        with open(part, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def download_one(fid):
    row = plan.loc[fid]
    expected, md5 = int(row["file_size"]), str(row["md5sum"]).lower()
    name, dest = row["file_name"], WSI_DIR / row["file_name"]
    part = PART_DIR / f"{fid}.part"

    if shutil.disk_usage(WSI_DIR).free < expected + DISK_MARGIN:
        raise DiskFull(f"Volume penuh, tidak muat {name} ({expected / 1e6:.0f} MB)")

    session, err, t0 = make_session(), "", time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > expected:
                part.unlink()
                have = 0
            if have < expected:
                if have:
                    log(f"   ~ lanjut {name} dari {have / 1e6:.0f} MB")
                stream_to_part(session, fid, part, have)
            size = part.stat().st_size
            if size != expected:
                raise IOError(f"ukuran belum lengkap ({size}/{expected})")
            if file_md5(part) != md5:
                part.unlink()
                raise ValueError("md5 tidak cocok (file korup)")
            ok, info = verify_svs(part)
            if not ok:
                part.unlink()
                raise NonRetryable(info)
            os.replace(part, dest)          # satu filesystem -> atomik & instan
            set_state(fid, "done", attempt)
            dt = max(time.time() - t0, 0.1)
            log(f"   OK  {name}  ({expected / 1e6:.0f} MB, {expected / 1e6 / dt:.1f} MB/s)")
            return
        except NonRetryable as e:
            err = str(e)
            break
        except (requests.RequestException, OSError, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
            log(f"   !   {name} percobaan {attempt}/{MAX_RETRIES}: {err}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * attempt)
    set_state(fid, "failed", MAX_RETRIES, err)
    log(f"   GAGAL {name}: {err}  -> diganti kandidat berikutnya")


def run_class(label, quota):
    while True:
        sub = plan[plan["label_out"] == label].sort_values("rank")
        need = quota - int((sub["status"] == "done").sum())
        if need <= 0:
            break
        todo = sub[sub["status"] == "pending"].head(need)
        if todo.empty:
            log(f"PERINGATAN: kandidat {label} habis; terkumpul {quota - need}/{quota}.")
            break
        log(f"\n[{label}] mengunduh {len(todo)} file (target {quota}, selesai {quota - need}) ...")
        ex = ThreadPoolExecutor(MAX_WORKERS)
        try:
            for f in as_completed([ex.submit(download_one, fid) for fid in todo.index]):
                f.result()
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)


# ---------------- MAIN ----------------
def main():
    global plan
    if not LABELED_CSV.exists():
        raise FileNotFoundError(f"{LABELED_CSV} tidak ada. Jalankan read_labels_luad.py dulu.")
    WSI_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    for p in WSI_DIR.glob("*.tmp"):
        p.unlink()

    plan = load_progress(build_plan())

    # file yang sudah ada dengan ukuran benar dianggap selesai
    for fid, r in plan[plan["status"] == "pending"].iterrows():
        dest = WSI_DIR / r["file_name"]
        if dest.exists() and dest.stat().st_size == int(r["file_size"]):
            plan.at[fid, "status"] = "done"
            plan.at[fid, "finished_at"] = "sudah ada"

    # buang .part yatim / milik file yang sudah selesai
    for p in PART_DIR.glob("*.part"):
        if p.stem not in plan.index or plan.at[p.stem, "status"] == "done":
            p.unlink(missing_ok=True)

    print("=" * 64)
    print("RENCANA DOWNLOAD (urut file terbesar)")
    print("=" * 64)
    need_bytes = 0
    for lab, quota in (("Mutant", N_MUTANT), ("Wildtype", N_WILDTYPE)):
        sel = plan[(plan["label_out"] == lab) & plan["selected"]]
        left = sel[sel["status"] != "done"]
        need_bytes += left["file_size"].sum()
        print(f"{lab:9s}: target {quota:3d} | dipilih {len(sel):3d} | {sel['file_size'].sum() / 1e9:6.1f} GB "
              f"| min/median/max {sel['size_mb'].min():.0f}/{sel['size_mb'].median():.0f}/{sel['size_mb'].max():.0f} MB "
              f"| selesai {int((sel['status'] == 'done').sum())}")
    print(f"Sisa yang perlu diunduh: ~{need_bytes / 1e9:.1f} GB")
    plan.to_csv(PLAN_CSV)
    free = shutil.disk_usage(WSI_DIR).free
    print(f"Ruang kosong di volume : {free / 1e9:.1f} GB")
    if need_bytes > free - DISK_MARGIN:
        print("PERINGATAN: volume tidak cukup untuk semua; unduhan berhenti otomatis saat penuh.")
    if DRY_RUN:
        print("\nDRY_RUN aktif: berhenti sebelum download.")
        return

    try:
        run_class("Mutant", N_MUTANT)
        run_class("Wildtype", N_WILDTYPE)
    except DiskFull as e:
        print(f"\nBERHENTI: {e}")
    except KeyboardInterrupt:
        print("\nDihentikan manual. Jalankan ulang untuk melanjutkan.")
    finally:
        with _lock:
            save_progress()
            write_labels()

    print("\n" + "=" * 64)
    print("LAPORAN AKHIR")
    print("=" * 64)
    for lab, quota in (("Mutant", N_MUTANT), ("Wildtype", N_WILDTYPE)):
        d = plan[(plan["label_out"] == lab) & (plan["status"] == "done")]
        print(f"{lab:9s}: {len(d)}/{quota} file | {d['file_size'].sum() / 1e9:.1f} GB")
    failed = plan[plan["status"] == "failed"]
    if len(failed):
        print(f"\nGagal permanen ({len(failed)}):")
        print(failed[["file_name", "error"]].to_string())
    print(f"\nFile: {WSI_DIR}\nLabel: {LABELS_CSV}\nProgres: {PROGRESS_CSV}")


if __name__ == "__main__":
    main()
