# Final Test Evaluation — ABMIL vs CLAM

- TEST: **40 WSI** (10 Mutant, 30 Wild-Type)
- Locked ABMIL threshold: **0.291687**
- Locked CLAM threshold: **0.166404**
- Probabilitas final: mean ensemble dari 5 checkpoint fold.

| Metrik | ABMIL | CLAM | Δ CLAM−ABMIL |
|---|---:|---:|---:|
| auroc | 0.7700 | 0.7733 | +0.0033 |
| auprc | 0.5774 | 0.6697 | +0.0924 |
| sensitivity | 0.8000 | 0.9000 | +0.1000 |
| specificity | 0.4000 | 0.3667 | -0.0333 |
| accuracy | 0.5000 | 0.5000 | +0.0000 |
| balanced_accuracy | 0.6000 | 0.6333 | +0.0333 |
| precision | 0.3077 | 0.3214 | +0.0137 |
| npv | 0.8571 | 0.9167 | +0.0595 |
| f1 | 0.4444 | 0.4737 | +0.0292 |
| mcc | 0.1816 | 0.2520 | +0.0704 |

ABMIL AUROC 95% bootstrap CI: **0.573–0.933**
CLAM AUROC 95% bootstrap CI: **0.577–0.933**

> TEST sudah dibuka. Jangan melakukan tuning model/threshold berdasarkan hasil ini. Gunakan final_test_paired_predictions.csv untuk DeLong dan McNemar final.
