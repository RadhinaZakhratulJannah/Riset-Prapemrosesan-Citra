#!/usr/bin/env python3
"""
save_research_results.py
Menyimpan hasil penelitian TCGA-LUAD secara bertahap tanpa training ulang.

Tahap:
1_init
2_dataset
3_preprocessing
4_features
5_abmil
6_clam
7_comparison
8_final_test
9_code
10_package
all
"""

import argparse, csv, hashlib, json, platform, shutil, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path("/workspace/TCGA-LUAD")
CODE = Path("/workspace/code")
TEXT_EXT = {".csv",".json",".txt",".md",".log",".yaml",".yml"}

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def human_size(n):
    v=float(n)
    for u in ["B","KB","MB","GB","TB"]:
        if v < 1024 or u=="TB":
            return f"{v:.2f} {u}"
        v/=1024

def run_capture(cmd):
    try:
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
        return r.stdout
    except Exception as e:
        return f"ERROR: {e}"

def ensure(dest):
    names=[
        "00_PROJECT_INFO","01_DATASET_SPLIT","02_PREPROCESSING",
        "03_FEATURE_EXTRACTION","04_ABMIL","05_CLAM",
        "06_COMPARISON_OOF","07_STATISTICAL_TESTS_OOF",
        "08_FINAL_TEST","09_CODE_SNAPSHOT","10_LOGS","11_MANIFEST"
    ]
    for n in names:
        (dest/n).mkdir(parents=True,exist_ok=True)

def log_stage(dest,stage,msgs):
    p=dest/"10_LOGS"/f"{stage}.txt"
    with open(p,"a",encoding="utf-8") as f:
        f.write(f"\n===== {now()} =====\n")
        for m in msgs:
            f.write(m+"\n")

def copy_file(src,dst_dir,msgs,max_mb=100):
    src=Path(src)
    if not src.exists() or not src.is_file():
        msgs.append(f"[SKIP] Tidak ditemukan: {src}")
        return
    mb=src.stat().st_size/(1024**2)
    if mb>max_mb:
        msgs.append(f"[SKIP BESAR] {src} ({mb:.1f} MB)")
        return
    dst_dir.mkdir(parents=True,exist_ok=True)
    dst=dst_dir/src.name
    shutil.copy2(src,dst)
    msgs.append(f"[OK] {src} -> {dst} ({human_size(dst.stat().st_size)})")

def copy_tree_small(src,dst,msgs,checkpoints=False):
    src=Path(src)
    if not src.exists():
        msgs.append(f"[SKIP] Folder tidak ditemukan: {src}")
        return
    count=0
    for p in src.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        allow=p.suffix.lower() in TEXT_EXT
        if checkpoints and p.name=="best_model.pt":
            allow=True
        if not allow:
            continue
        if p.stat().st_size > 100*1024**2:
            continue
        q=dst/p.relative_to(src)
        q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,q)
        count+=1
    msgs.append(f"[INFO] {count} file disalin dari {src}")

def stage1(root,code,dest):
    msgs=[]
    info={
        "created_at":now(),
        "project_root":str(root),
        "code_dir":str(code),
        "python":sys.version,
        "platform":platform.platform(),
        "hostname":platform.node()
    }
    (dest/"00_PROJECT_INFO"/"project_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    (dest/"00_PROJECT_INFO"/"gpu_info.txt").write_text(run_capture(["nvidia-smi"]),encoding="utf-8")
    (dest/"00_PROJECT_INFO"/"disk_info.txt").write_text(run_capture(["df","-h","/workspace"]),encoding="utf-8")
    (dest/"README_RESULTS.md").write_text(
        "# REPORT_RESULTS\n\nFolder hasil penelitian TCGA-LUAD yang dikumpulkan secara bertahap.\n",
        encoding="utf-8"
    )
    msgs += ["[OK] Struktur folder hasil dibuat","[OK] project_info.json","[OK] gpu_info.txt","[OK] disk_info.txt"]
    return msgs

def stage2(root,dest):
    msgs=[]
    out=dest/"01_DATASET_SPLIT"
    copy_file(root/"cv_folds_mutant_priority.csv",out,msgs)
    for p in root.glob("*.csv"):
        if p.name=="cv_folds_mutant_priority.csv":
            continue
        if any(k in p.name.lower() for k in ["label","dataset","split","fold","metadata"]):
            copy_file(p,out,msgs,50)
    return msgs

def stage3(root,dest):
    msgs=[]
    out=dest/"02_PREPROCESSING"
    for dname in ["Patches","Patches_Macenko","PRE PROSECING","PREPROCESSING"]:
        d=root/dname
        if not d.exists():
            msgs.append(f"[SKIP] Folder tidak ditemukan: {d}")
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in TEXT_EXT:
                if any(k in p.name.lower() for k in ["summary","manifest","qc","report","stat"]):
                    copy_file(p,out/dname,msgs,50)
    msgs.append("[INFO] Patch/citra besar tidak disalin.")
    return msgs

def stage4(root,dest):
    msgs=[]
    out=dest/"03_FEATURE_EXTRACTION"
    copy_file(root/"Features_Swin"/"features_summary.csv",out,msgs)
    feat=root/"Features_Swin"
    if feat.exists():
        for p in feat.iterdir():
            if p.is_file() and p.suffix.lower() in TEXT_EXT and p.name!="features_summary.csv":
                copy_file(p,out,msgs,50)
    msgs.append("[INFO] pt_files/h5_files tidak disalin.")
    return msgs

def stage5(root,dest):
    msgs=[]
    copy_tree_small(root/"ABMIL_MutantPriority",dest/"04_ABMIL",msgs,checkpoints=True)
    for p in root.glob("*abmil*.log"):
        copy_file(p,dest/"04_ABMIL"/"logs",msgs)
    return msgs

def stage6(root,dest):
    msgs=[]
    copy_tree_small(root/"CLAM_SB_MutantPriority_Final",dest/"05_CLAM",msgs,checkpoints=True)
    for p in root.glob("*clam*.log"):
        copy_file(p,dest/"05_CLAM"/"logs",msgs)
    return msgs

def stage7(root,dest):
    msgs=[]
    copy_tree_small(root/"ABMIL_vs_CLAM_Comparison",dest/"06_COMPARISON_OOF",msgs)
    copy_tree_small(root/"ABMIL_vs_CLAM_StatTests",dest/"07_STATISTICAL_TESTS_OOF",msgs)
    return msgs

def stage8(root,dest):
    msgs=[]
    copy_tree_small(root/"Final_Test_ABMIL_vs_CLAM",dest/"08_FINAL_TEST",msgs)
    for p in root.glob("*final*test*.log"):
        copy_file(p,dest/"08_FINAL_TEST"/"logs",msgs)
    return msgs

def stage9(code,dest):
    msgs=[]
    out=dest/"09_CODE_SNAPSHOT"
    if not code.exists():
        return [f"[SKIP] Folder kode tidak ditemukan: {code}"]
    for p in code.iterdir():
        if p.is_file() and (p.suffix.lower()==".py" or p.name in ["README.md","requirements.txt",".gitignore"]):
            copy_file(p,out,msgs,20)
    (out/"requirements_snapshot.txt").write_text(run_capture([sys.executable,"-m","pip","freeze"]),encoding="utf-8")
    msgs.append("[OK] requirements_snapshot.txt")
    return msgs

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def stage10(root,dest):
    msgs=[]
    manifest=dest/"11_MANIFEST"/"file_manifest.csv"
    rows=[]
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p!=manifest:
            rows.append({
                "relative_path":str(p.relative_to(dest)),
                "size_bytes":p.stat().st_size,
                "size_human":human_size(p.stat().st_size),
                "sha256":sha256(p)
            })
    with open(manifest,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["relative_path","size_bytes","size_human","sha256"])
        w.writeheader(); w.writerows(rows)
    stamp=datetime.now().strftime("%Y%m%d_%H%M")
    zpath=root/f"REPORT_RESULTS_{stamp}.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in dest.rglob("*"):
            if p.is_file():
                z.write(p,p.relative_to(root))
    msgs += [f"[OK] Manifest: {manifest}",f"[OK] ZIP: {zpath} ({human_size(zpath.stat().st_size)})"]
    return msgs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--code-dir",type=Path,default=CODE)
    ap.add_argument("--dest",type=Path,default=None)
    ap.add_argument("--stage",required=True,choices=[
        "1_init","2_dataset","3_preprocessing","4_features","5_abmil",
        "6_clam","7_comparison","8_final_test","9_code","10_package","all"
    ])
    a=ap.parse_args()
    dest=a.dest or (a.root/"REPORT_RESULTS")
    ensure(dest)

    stages=["1_init","2_dataset","3_preprocessing","4_features","5_abmil","6_clam","7_comparison","8_final_test","9_code","10_package"] if a.stage=="all" else [a.stage]

    for s in stages:
        if s=="1_init": msgs=stage1(a.root,a.code_dir,dest)
        elif s=="2_dataset": msgs=stage2(a.root,dest)
        elif s=="3_preprocessing": msgs=stage3(a.root,dest)
        elif s=="4_features": msgs=stage4(a.root,dest)
        elif s=="5_abmil": msgs=stage5(a.root,dest)
        elif s=="6_clam": msgs=stage6(a.root,dest)
        elif s=="7_comparison": msgs=stage7(a.root,dest)
        elif s=="8_final_test": msgs=stage8(a.root,dest)
        elif s=="9_code": msgs=stage9(a.code_dir,dest)
        elif s=="10_package": msgs=stage10(a.root,dest)
        else: raise RuntimeError(s)

        print(f"\n===== {s} =====")
        for m in msgs:
            print(m)
        log_stage(dest,s,msgs)

    print(f"\nSELESAI. Hasil: {dest}")

if __name__=="__main__":
    main()
