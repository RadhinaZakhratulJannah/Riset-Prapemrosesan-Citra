"""Tahap 1: gabungkan label EGFR (Excel) dengan daftar slide DX GDC -> labeled_all_dx.csv.
Versi RunPod: semua file ada di /workspace/TCGA-LUAD (tanpa Google Drive/Colab).

Input yang harus ada di BASE_DIR:
  EGFR.xlsx
  gdc_check_output/gdc_dx_slides_all.csv
"""
import os
from pathlib import Path
import pandas as pd

# ---------------- KONFIGURASI ----------------
BASE_DIR = Path(os.environ.get("TCGA_BASE", "/workspace/TCGA-LUAD"))
EXCEL_PATH = BASE_DIR / "EGFR.xlsx"
OUT_DIR = BASE_DIR / "gdc_check_output"
DX_CSV = OUT_DIR / "gdc_dx_slides_all.csv"
ID_COL = "Sample ID"

TARGET_MAG = 40
ONLY_PRIMARY_TUMOR = True
INCLUDE_WILDTYPE = False        # manifest ini hanya laporan; skrip download memilih sendiri dari labeled_all_dx.csv
INCLUDE_UNREADABLE_MAG = False


def pid(s):
    return str(s).strip().upper()[:12]      # TCGA-XX-XXXX


def join_unique(series):
    return "; ".join(dict.fromkeys(str(x) for x in series.dropna()))


def check_inputs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in (EXCEL_PATH, DX_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "File input belum ada di Pod:\n  " + "\n  ".join(map(str, missing)) +
            "\nUpload lewat JupyterLab, atau tarik dari Drive dengan rclone (lihat setup.sh).")
    print(f"[OK] BASE_DIR = {BASE_DIR}")


def main():
    check_inputs()

    # ---------------- 1) BACA LABEL DARI EXCEL ----------------
    egfr = pd.read_excel(EXCEL_PATH)          # butuh openpyxl
    egfr["patient_id"] = egfr[ID_COL].map(pid)

    agg = {"n_mutations": ("patient_id", "size")}
    if "Protein Change" in egfr.columns:
        agg["protein_change"] = ("Protein Change", join_unique)
    if "Mutation Type" in egfr.columns:
        agg["mutation_type"] = ("Mutation Type", join_unique)
    labels = egfr.groupby("patient_id").agg(**agg).reset_index()
    mutant_ids = set(labels["patient_id"])

    print("=" * 62)
    print("1) LABEL DARI EXCEL")
    print("=" * 62)
    print(f"Baris: {len(egfr)} | pasien unik (Mutant): {len(mutant_ids)}")
    print(f"Pasien dengan >1 mutasi EGFR: {(labels['n_mutations'] > 1).sum()}")
    if "Mutation Type" in egfr.columns:
        print("\nJenis mutasi (per baris):")
        print(egfr["Mutation Type"].value_counts().to_string())
    if "Protein Change" in egfr.columns:
        print("\nPerubahan protein terbanyak (per baris):")
        print(egfr["Protein Change"].value_counts().head(8).to_string())

    # ---------------- 2) GABUNGKAN DENGAN SLIDE DX ----------------
    dx = pd.read_csv(DX_CSV)
    dx["patient_id"] = dx["patient_id"].map(pid)
    dx = dx.merge(labels, on="patient_id", how="left")
    dx["label"] = dx["patient_id"].isin(mutant_ids).map({True: "Mutant", False: "Wildtype_unverified"})

    is_primary = dx["sample_type"].isna() | dx["sample_type"].eq("Primary Tumor")
    mag = pd.to_numeric(dx["magnification"], errors="coerce")
    mag_ok = mag.eq(TARGET_MAG)
    mag_unknown = mag.isna()

    def reason(i):
        if ONLY_PRIMARY_TUMOR and not is_primary[i]:
            return "bukan tumor primer"
        if mag_unknown[i] and not INCLUDE_UNREADABLE_MAG:
            return "magnifikasi tidak terbaca"
        if not mag_unknown[i] and not mag_ok[i]:
            return f"magnifikasi {int(mag[i])}x"
        if dx.at[i, "label"] != "Mutant" and not INCLUDE_WILDTYPE:
            return "bukan pasien di Excel"
        return "terpilih"

    dx["seleksi"] = [reason(i) for i in dx.index]
    sel = dx[dx["seleksi"] == "terpilih"].copy()
    gb = lambda d: d["file_size"].sum() / 1e9

    print("\n" + "=" * 62)
    print(f"2) SLIDE DX ({len(dx)} file, {dx['patient_id'].nunique()} pasien)")
    print("=" * 62)
    print(dx["seleksi"].value_counts().to_string())

    cand = dx[(mag_ok | (mag_unknown & INCLUDE_UNREADABLE_MAG)) & (is_primary | (not ONLY_PRIMARY_TUMOR))]
    print(f"\nSlide DX {TARGET_MAG}x (semua pasien) per label:")
    for lab, g in cand.groupby("label"):
        print(f"  {lab:20s}: {len(g):4d} file | {g['patient_id'].nunique():4d} pasien | {gb(g):.1f} GB")

    # ---------------- 3) DIAGNOSTIK PASIEN EXCEL ----------------
    print("\n" + "=" * 62)
    print("3) STATUS TIAP PASIEN MUTANT (di Excel)")
    print("=" * 62)
    have = dx.groupby("patient_id")
    cats = {"terpilih (ada DX 40x)": [], "hanya DX 20x": [], "DX magnifikasi tidak terbaca": [],
            "tidak punya DX di GDC": []}
    for p in sorted(mutant_ids):
        if p not in have.groups:
            cats["tidak punya DX di GDC"].append(p)
            continue
        m = pd.to_numeric(have.get_group(p)["magnification"], errors="coerce")
        if m.eq(TARGET_MAG).any():
            cats["terpilih (ada DX 40x)"].append(p)
        elif m.isna().any():
            cats["DX magnifikasi tidak terbaca"].append(p)
        else:
            cats["hanya DX 20x"].append(p)
    for k, v in cats.items():
        print(f"{k:32s}: {len(v)}")
        if v and not k.startswith("terpilih"):
            print("    ", ", ".join(v))

    # ---------------- 4) HASIL AKHIR ----------------
    print("\n" + "=" * 62)
    print(f"4) MANIFEST (INCLUDE_WILDTYPE={INCLUDE_WILDTYPE})")
    print("=" * 62)
    print(f"Total: {len(sel)} file | {sel['patient_id'].nunique()} pasien | {gb(sel):.1f} GB")
    for lab, g in sel.groupby("label"):
        print(f"  {lab:20s}: {len(g):4d} file | {g['patient_id'].nunique():4d} pasien | {gb(g):.1f} GB")
    per_pat = sel.groupby("patient_id").size()
    print(f"Pasien dengan >1 slide DX: {(per_pat > 1).sum()} (split train/test harus per pasien)")

    keep = ["file_id", "file_name", "file_size", "md5sum", "patient_id", "label", "magnification",
            "n_mutations", "protein_change", "mutation_type"]
    keep = [c for c in keep if c in sel.columns]
    dx.to_csv(OUT_DIR / "labeled_all_dx.csv", index=False)
    sel[keep].to_csv(OUT_DIR / "labeled_manifest.csv", index=False)
    print(f"\nTersimpan di {OUT_DIR}: labeled_all_dx.csv, labeled_manifest.csv")


if __name__ == "__main__":
    main()
