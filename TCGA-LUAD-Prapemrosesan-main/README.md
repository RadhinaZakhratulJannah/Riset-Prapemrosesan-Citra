# TCGA-LUAD EGFR Mutation Prediction

Prediksi status mutasi **EGFR (Mutant vs Wild-Type)** pada citra histopatologi **Lung Adenocarcinoma (LUAD)** menggunakan **Swin Transformer Tiny**, **Attention-Based Multiple Instance Learning (ABMIL)**, dan **CLAM-SB**.

Repository ini berisi source code, konfigurasi eksperimen, dataset split, hasil cross-validation, evaluasi final, dan uji statistik yang digunakan dalam penelitian skripsi.

---

## Research Overview

### Objective

Penelitian ini bertujuan memprediksi status mutasi EGFR pada Whole Slide Image (WSI) histopatologi LUAD dengan pendekatan weakly supervised learning.

Dua metode Multiple Instance Learning dibandingkan:

- **Swin-Tiny + ABMIL**
- **Swin-Tiny + CLAM-SB**

Evaluasi dilakukan menggunakan:

- 5-Fold Patient-Level Cross Validation
- Independent Final Test
- AUROC
- AUPRC
- Sensitivity
- Specificity
- Balanced Accuracy
- F1-score
- MCC
- DeLong Test
- McNemar Test

---

## Research Pipeline

```text
TCGA-LUAD WSI
       |
       v
Tissue Detection
       |
       v
Patch Extraction
224 x 224 pixels
       |
       v
Quality Control
       |
       v
Macenko Color Normalization
       |
       v
Swin Transformer Tiny
Feature Extraction
       |
       v
768-dimensional Patch Features
       |
       +----------------------+
       |                      |
       v                      v
     ABMIL                  CLAM-SB
       |                      |
       +----------+-----------+
                  |
                  v
       5-Fold Cross Validation
                  |
                  v
          OOF Prediction
                  |
                  v
     Mutant-Priority Threshold
                  |
                  v
       Independent Final Test
                  |
                  v
        DeLong + McNemar
```

---

## Dataset

Dataset berasal dari **TCGA-LUAD**.

| Dataset | Total | Mutant | Wild-Type |
|---|---:|---:|---:|
| Development / Cross Validation | 160 | 40 | 120 |
| Independent Final Test | 40 | 10 | 30 |
| **Total** | **200** | **50** | **150** |

Kriteria utama dataset:

- Project: TCGA-LUAD
- Histopathology WSI
- Format: SVS
- Diagnostic slide
- Satu WSI per pasien
- Label: EGFR Mutant atau Wild-Type

---

## Preprocessing

Tahapan preprocessing meliputi:

### 1. Tissue Detection

Area jaringan dideteksi sebelum patch extraction untuk mengurangi background kosong.

Program:

```text
src/preprocessing/tissue_detection_luad.py
```

### 2. Patch Extraction

WSI dibagi menjadi patch berukuran:

```text
224 x 224 pixels
```

Program:

```text
src/preprocessing/patch_extraction_luad.py
```

Quality control mencatat:

- `mask_coverage`
- `tissue_fraction`
- `blur_score`

### Patch Extraction Summary

| Information | Result |
|---|---:|
| Total WSI | 200 |
| Total Patch | 11,551,006 |
| Mean Patch / WSI | 57,755.03 |
| Minimum Patch / WSI | 160 |
| Maximum Patch / WSI | 165,616 |
| Original Manifest Size | 2.167 GB |

Summary tersedia pada:

```text
results/preprocessing/
```

### 3. Macenko Color Normalization

Normalisasi warna digunakan untuk mengurangi variasi pewarnaan H&E antar-slide.

Program:

```text
src/preprocessing/macenko_luad.py
```

---

## Feature Extraction

Feature extraction dilakukan menggunakan:

**Swin Transformer Tiny**

Setiap patch diubah menjadi feature vector berdimensi:

```text
768
```

Program:

```text
src/features/swin_feature_extraction_luad.py
```

Summary hasil feature extraction tersedia pada:

```text
results/feature_extraction/features_summary.csv
```

---

# Multiple Instance Learning

## ABMIL

ABMIL memberikan nilai attention pada setiap patch feature.

Patch dengan informasi yang lebih relevan mendapatkan bobot attention lebih besar sebelum seluruh feature digabung menjadi representasi WSI.

Program utama:

```text
src/mil/abmil_mutant_priority.py
```

---

## CLAM-SB

CLAM-SB menggunakan attention-based pooling dan instance-level clustering.

Metode ini membantu model mempelajari patch yang memiliki kontribusi tinggi dan rendah terhadap keputusan pada level WSI.

Program utama:

```text
src/mil/clam_mutant_priority_final.py
```

---

# 5-Fold Cross Validation

Cross validation dilakukan pada **160 WSI development dataset**.

Setiap fold memiliki:

```text
Training:
96 Wild-Type
32 Mutant

Validation:
24 Wild-Type
8 Mutant
```

Pembagian dilakukan secara patient-level untuk mencegah data pasien yang sama muncul pada training dan validation.

File pembagian fold:

```text
results/dataset_split/cv_folds_mutant_priority.csv
```

---

# OOF Evaluation Results

OOF = Out-of-Fold prediction dari seluruh validation fold.

## ABMIL OOF

| Metric | Result |
|---|---:|
| AUROC | 0.7283 |
| AUPRC | 0.5590 |
| Sensitivity | 0.8000 |
| Specificity | 0.5167 |
| Accuracy | 0.5875 |
| Balanced Accuracy | 0.6583 |
| F1-score | 0.492 |
| MCC | 0.276 |

Locked mutant-priority threshold:

```text
0.291687
```

---

## CLAM-SB OOF

| Metric | Result |
|---|---:|
| AUROC | 0.7135 |
| AUPRC | 0.5414 |
| Sensitivity | 0.8000 |
| Specificity | 0.5250 |
| Accuracy | 0.5938 |
| Balanced Accuracy | 0.6625 |
| F1-score | 0.496 |
| MCC | 0.283 |

Locked mutant-priority threshold:

```text
0.166404
```

---

# OOF Model Comparison

| Metric | ABMIL | CLAM-SB |
|---|---:|---:|
| AUROC | **0.7283** | 0.7135 |
| AUPRC | **0.5590** | 0.5414 |
| Sensitivity | 0.8000 | 0.8000 |
| Specificity | 0.5167 | **0.5250** |
| Balanced Accuracy | 0.6583 | **0.6625** |
| F1-score | 0.492 | **0.496** |
| MCC | 0.276 | **0.283** |

---

# OOF Statistical Tests

## DeLong Test

Perbandingan AUROC:

```text
ABMIL AUROC = 0.7283
CLAM AUROC  = 0.7135

Delta       = +0.0148
p-value     = 0.602610
```

Interpretation:

```text
No statistically significant difference.
```

---

## McNemar Test

Pada mutant-priority threshold:

```text
ABMIL threshold = 0.291687
CLAM threshold  = 0.166404

Exact p-value   = 1.000000
```

Interpretation:

```text
No statistically significant difference.
```

---

# Independent Final Test

Final evaluation dilakukan pada **40 WSI yang tidak digunakan dalam proses training maupun cross validation**.

Komposisi:

```text
10 Mutant
30 Wild-Type
```

Threshold yang digunakan telah dikunci berdasarkan development data sebelum final test.

---

## ABMIL Final Test

| Metric | Result |
|---|---:|
| AUROC | 0.7700 |
| AUPRC | 0.5774 |
| Sensitivity | 0.8000 |
| Specificity | 0.4000 |
| Accuracy | 0.5000 |
| Balanced Accuracy | 0.6000 |
| Precision | 0.3077 |
| NPV | 0.8571 |
| F1-score | 0.4444 |
| MCC | 0.1816 |

Confusion Matrix:

```text
TP = 8
TN = 12
FP = 18
FN = 2
```

---

## CLAM-SB Final Test

| Metric | Result |
|---|---:|
| AUROC | 0.7733 |
| AUPRC | 0.6697 |
| Sensitivity | 0.9000 |
| Specificity | 0.3667 |
| Accuracy | 0.5000 |
| Balanced Accuracy | 0.6333 |
| Precision | 0.3214 |
| NPV | 0.9167 |
| F1-score | 0.4737 |
| MCC | 0.2520 |

Confusion Matrix:

```text
TP = 9
TN = 11
FP = 19
FN = 1
```

---

# Final Model Comparison

| Metric | ABMIL | CLAM-SB |
|---|---:|---:|
| AUROC | 0.7700 | **0.7733** |
| AUPRC | 0.5774 | **0.6697** |
| Sensitivity | 0.8000 | **0.9000** |
| Specificity | **0.4000** | 0.3667 |
| Balanced Accuracy | 0.6000 | **0.6333** |
| F1-score | 0.4444 | **0.4737** |
| MCC | 0.1816 | **0.2520** |

CLAM-SB menghasilkan sensitivity lebih tinggi pada independent test, sedangkan ABMIL menghasilkan specificity sedikit lebih tinggi.

---

# Final Statistical Tests

## DeLong Test

```text
ABMIL AUROC = 0.7700
CLAM AUROC  = 0.7733

Delta       = -0.0033

95% CI:
-0.0499 to 0.0432

p-value = 0.888434
```

Interpretation:

```text
The AUROC difference is not statistically significant.
```

---

## McNemar Test

```text
Both Correct               = 19
ABMIL Correct / CLAM Wrong = 1
ABMIL Wrong / CLAM Correct = 1
Both Wrong                 = 19

Discordant Pairs = 2
Exact p-value    = 1.000000
```

Interpretation:

```text
The difference in paired classification results is not statistically significant.
```

---

# Repository Structure

```text
TCGA-LUAD-SWIN-ABMIL-CLAM/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   │
│   ├── data/
│   │   ├── read_labels_luad.py
│   │   ├── download_wsi_luad.py
│   │   └── split_dataset_luad.py
│   │
│   ├── preprocessing/
│   │   ├── tissue_detection_luad.py
│   │   ├── patch_extraction_luad.py
│   │   └── macenko_luad.py
│   │
│   ├── features/
│   │   └── swin_feature_extraction_luad.py
│   │
│   ├── mil/
│   │   ├── abmil_mutant_priority.py
│   │   └── clam_mutant_priority_final.py
│   │
│   ├── evaluation/
│   │   ├── compare_abmil_clam.py
│   │   ├── delong_mcnemar_abmil_vs_clam.py
│   │   ├── final_test_evaluation_abmil_clam.py
│   │   └── final_test_delong_mcnemar.py
│   │
│   └── utilities/
│       └── save_research_results.py
│
└── results/
    ├── dataset_split/
    ├── preprocessing/
    ├── feature_extraction/
    ├── abmil/
    ├── clam/
    ├── comparison_oof/
    ├── statistics_oof/
    └── final_test/
```

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/Mfajrialfaini/TCGA-LUAD-SWIN-ABMIL-CLAM.git
```

```bash
cd TCGA-LUAD-SWIN-ABMIL-CLAM
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

```bash
python src/data/read_labels_luad.py
```

```bash
python src/data/download_wsi_luad.py
```

```bash
python src/data/split_dataset_luad.py
```

---

## 4. Tissue Detection

```bash
python src/preprocessing/tissue_detection_luad.py
```

---

## 5. Patch Extraction

```bash
python src/preprocessing/patch_extraction_luad.py
```

---

## 6. Macenko Normalization

```bash
python src/preprocessing/macenko_luad.py
```

---

## 7. Swin-Tiny Feature Extraction

```bash
python src/features/swin_feature_extraction_luad.py
```

---

## 8. Train ABMIL

```bash
python src/mil/abmil_mutant_priority.py
```

---

## 9. Train CLAM-SB

```bash
python src/mil/clam_mutant_priority_final.py
```

---

## 10. Compare OOF Results

```bash
python src/evaluation/compare_abmil_clam.py
```

---

## 11. OOF Statistical Tests

```bash
python src/evaluation/delong_mcnemar_abmil_vs_clam.py
```

---

## 12. Independent Final Test

```bash
python src/evaluation/final_test_evaluation_abmil_clam.py
```

---

## 13. Final Statistical Tests

```bash
python src/evaluation/final_test_delong_mcnemar.py
```

---

# Large Data

Karena ukurannya sangat besar, file berikut tidak disimpan pada GitHub:

```text
*.svs
Patches/
Patches_Macenko/
Features_Swin/pt_files/
Features_Swin/h5_files/
*.h5
*.pt
```

Repository ini terutama menyimpan:

- Source code
- Dataset split
- Preprocessing summary
- Feature extraction summary
- Training configuration
- OOF predictions
- Evaluation metrics
- Statistical test results
- Independent final test results

---

# Reproducibility Note

Independent final test telah digunakan sebagai evaluasi akhir.

Oleh karena itu:

> Threshold, model checkpoint, dan hyperparameter tidak boleh disesuaikan berdasarkan hasil final test.

Hal ini dilakukan untuk menjaga independensi dan validitas evaluasi akhir.

---

# Author

**Muhammad Fajri Alfaini**

Research:

**EGFR Mutation Prediction in Lung Adenocarcinoma Histopathology Using Swin Transformer and Multiple Instance Learning**

Methods:

```text
Swin Transformer Tiny + ABMIL
Swin Transformer Tiny + CLAM-SB
```

Repository:

```text
https://github.com/Mfajrialfaini/TCGA-LUAD-SWIN-ABMIL-CLAM
```
