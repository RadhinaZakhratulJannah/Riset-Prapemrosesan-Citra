# Uji Statistik ABMIL vs CLAM

Analisis default menggunakan **OOF 5-fold CV (development)**, bukan TEST.

## 1. DeLong test — AUROC

| Item | Hasil |
|---|---:|
| N | 160 |
| Mutant | 40 |
| Wild-Type | 120 |
| AUROC ABMIL | 0.7283 |
| AUROC CLAM | 0.7135 |
| Selisih AUROC (ABMIL−CLAM) | +0.0148 |
| 95% CI selisih | -0.0409 s.d. 0.0705 |
| z | 0.5207 |
| p-value DeLong | 0.602610 |
| Keputusan α=0.05 | tidak signifikan |

## 2. McNemar — threshold 0.5

| Item | Hasil |
|---|---:|
| ABMIL benar, CLAM salah | 9 |
| ABMIL salah, CLAM benar | 25 |
| Discordant pairs | 34 |
| Exact McNemar p | 0.009041 |
| Continuity-corrected p | 0.010097 |
| Keputusan α=0.05 | signifikan |

## 3. McNemar — threshold Mutant-Priority

- Threshold ABMIL: **0.291687**
- Threshold CLAM: **0.166404**

| Item | Hasil |
|---|---:|
| ABMIL benar, CLAM salah | 10 |
| ABMIL salah, CLAM benar | 10 |
| Discordant pairs | 20 |
| Exact McNemar p | 1.000000 |
| Continuity-corrected p | 0.823063 |
| Keputusan α=0.05 | tidak signifikan |

## 4. Exploratory — per kelas pada threshold Mutant-Priority

| Populasi | N | Exact McNemar p | Keterangan |
|---|---:|---:|---|
| Mutant saja | 40 | 1.000000 | Perbedaan sensitivitas secara paired |
| Wild-Type saja | 120 | 1.000000 | Perbedaan spesifisitas secara paired |

## Interpretasi

- DeLong menguji apakah perbedaan **AUROC** ABMIL dan CLAM lebih besar daripada yang diharapkan akibat variasi sampel.
- McNemar menguji apakah kedua model memiliki perbedaan pada **hasil klasifikasi benar/salah untuk slide yang sama**.
- p < 0.05 berarti terdapat bukti perbedaan statistik pada pengujian tersebut; p ≥ 0.05 berarti data ini belum memberikan bukti yang cukup untuk menyatakan perbedaan statistik.
- Karena analisis default menggunakan OOF CV dan threshold juga dikembangkan dari OOF, hasil ini sebaiknya diperlakukan sebagai analisis development. Uji final sebaiknya diulang pada independent TEST setelah semua pilihan model dan threshold dikunci.
