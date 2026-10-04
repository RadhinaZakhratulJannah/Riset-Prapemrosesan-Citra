# Perbandingan Swin-Tiny + ABMIL vs Swin-Tiny + CLAM

## OOF pada threshold Mutant-Priority

| Metrik | ABMIL | CLAM | Δ CLAM−ABMIL |
|---|---:|---:|---:|
| AUROC | 0.7283 | 0.7135 | -0.0148 |
| AUPRC | 0.5590 | 0.5414 | -0.0176 |
| Sensitivity | 0.8000 | 0.8000 | +0.0000 |
| Specificity | 0.5167 | 0.5250 | +0.0083 |
| Accuracy | 0.5875 | 0.5938 | +0.0062 |
| Balanced Accuracy | 0.6583 | 0.6625 | +0.0042 |
| Precision / PPV | 0.3556 | 0.3596 | +0.0040 |
| NPV | 0.8857 | 0.8873 | +0.0016 |
| F1-score | 0.4923 | 0.4961 | +0.0038 |
| MCC | 0.2764 | 0.2833 | +0.0068 |
| Threshold development | 0.291687 | 0.166404 | -0.125282 |
| AUROC 95% CI | 0.637–0.816 | 0.611–0.808 | — |

## Mean ± SD 5-fold pada threshold 0.5

| Metrik | ABMIL | CLAM | Δ mean CLAM−ABMIL |
|---|---:|---:|---:|
| AUROC | 0.7698 ± 0.1001 | 0.7115 ± 0.0631 | -0.0583 |
| AUPRC | 0.6264 ± 0.1459 | 0.5570 ± 0.1157 | -0.0694 |
| Sensitivity | 0.5750 ± 0.2271 | 0.5250 ± 0.1630 | -0.0500 |
| Specificity | 0.6500 ± 0.3112 | 0.8000 ± 0.1156 | +0.1500 |
| Accuracy | 0.6312 ± 0.1919 | 0.7312 ± 0.0927 | +0.1000 |
| Balanced Accuracy | 0.6125 ± 0.0961 | 0.6625 ± 0.0959 | +0.0500 |
| Precision / PPV | 0.4447 ± 0.2194 | 0.4915 ± 0.1428 | +0.0468 |
| NPV | 0.8146 ± 0.0757 | 0.8349 ± 0.0507 | +0.0203 |
| F1-score | 0.4521 ± 0.1086 | 0.4949 ± 0.1310 | +0.0428 |
| MCC | 0.2398 ± 0.2200 | 0.3240 ± 0.1837 | +0.0842 |

## AUROC dan AUPRC per fold

| Fold | ABMIL AUROC | CLAM AUROC | Δ AUROC | ABMIL AUPRC | CLAM AUPRC | Δ AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7552 | 0.6667 | -0.0885 | 0.5483 | 0.4442 | -0.1042 |
| 2 | 0.6771 | 0.6875 | +0.0104 | 0.4811 | 0.4454 | -0.0357 |
| 3 | 0.7135 | 0.6927 | -0.0208 | 0.6004 | 0.6020 | +0.0016 |
| 4 | 0.7656 | 0.6875 | -0.0781 | 0.6366 | 0.5750 | -0.0616 |
| 5 | 0.9375 | 0.8229 | -0.1146 | 0.8655 | 0.7184 | -0.1471 |

## Validasi data paired

- Jumlah OOF ABMIL: 160
- Jumlah OOF CLAM: 160
- Slide sama: True
- Label sama: True
- Fold sama: True
- Siap untuk uji paired berikutnya: True

> Catatan: tabel ini deskriptif. Signifikansi AUROC memerlukan uji DeLong, sedangkan hasil klasifikasi berpasangan dapat diuji dengan McNemar.
