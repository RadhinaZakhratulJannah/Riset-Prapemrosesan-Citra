# Final Statistical Tests — ABMIL vs CLAM

**Dataset:** independent FINAL TEST  
**N:** 40 (10 Mutant, 30 Wild-Type)  
**Threshold ABMIL locked:** 0.291687  
**Threshold CLAM locked:** 0.166404  

## 1. Paired DeLong test — AUROC

| Item | Result |
|---|---:|
| ABMIL AUROC | 0.7700 |
| CLAM AUROC | 0.7733 |
| Δ AUROC (ABMIL−CLAM) | -0.0033 |
| 95% CI Δ AUROC | -0.0499 to 0.0432 |
| z | -0.1403 |
| p-value | 0.888434 |
| Decision α=0.05 | **TIDAK SIGNIFIKAN** |

## 2. Exact McNemar test — locked thresholds

| Item | Result |
|---|---:|
| Both correct | 19 |
| ABMIL correct, CLAM wrong | 1 |
| ABMIL wrong, CLAM correct | 1 |
| Both wrong | 19 |
| Discordant pairs | 2 |
| Exact p-value | 1.000000 |
| Continuity-corrected p-value | 0.479500 |
| Decision α=0.05 | **TIDAK SIGNIFIKAN** |

## 3. Exploratory class-specific McNemar

| Class | N | Exact p-value |
|---|---:|---:|
| Mutant | 10 | 1.000000 |
| Wild-Type | 30 | 1.000000 |

## Interpretation rule

- p < 0.05: evidence of a statistically significant difference for that test.
- p ≥ 0.05: insufficient evidence of a statistically significant difference.
- DeLong evaluates AUROC differences using continuous probabilities.
- McNemar evaluates paired correct/incorrect classifications at the already locked thresholds.
- The class-specific tests are exploratory because the class sample sizes are small.
