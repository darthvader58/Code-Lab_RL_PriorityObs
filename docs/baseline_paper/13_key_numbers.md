# 13. Key Numbers and Where They Come From

Every number used in the paper, with its source. Update this file whenever the dataset is
regenerated. Dataset: `trackwise_decisions.parquet`, 351,639 × 40,
SHA-256 `4b92f01e24e231a5b5a1151060fef586f08e450e15c252fb23ef9ef772fd1466`.

## Dataset

| quantity | value | source |
|---|---|---|
| decisions (rows) | 351,639 | parquet; `task_outputs/final_validation_report.md` |
| columns | 40 | same |
| missing decision IDs / discontinuities | 1,161 / 49 | `../feature_set.md` |
| period (G5NR time) | June 2005 – January 2006 | `../trackwise_method.md` §4.4 |
| distinct 30-min weather states | 11,731 | `../trackwise_method.md` |
| model resolution | 0.0625° (~7 km), 30 min | G5NR |
| orbit altitude used / MSI swath | ~394 km / 150 km | `extract_trackwise.py` |
| nadir samples per decision | 60 (1 s) | `task_outputs/t4_land_water_report.md` |
| full-download size avoided | ~10.1 TB (9.2 TiB) | `../trackwise_method.md` §3 |
| corrupted readings removed | 710 rows (0.20%) | `../trackwise_method.md` §4.5 |
| invalid target entries / decisions | 94 / 51 | `final_validation_report.md` |
| finite corrupt targets set to NaN | 24 (15 decisions) | same |
| `lwtup_fore` finite / NaN | 339,859 / 11,780 | same |
| orbit-sampled land / water mean | 0.339 / 0.661 | `t4_land_water_report.md` |
| mixed-surface decisions (0.05–0.95 land) | 17.09% | same |
| CPR / ATLID power | ~308 W / ~310 W | `../feature_set.md` |

## Labels (computed for this paper, threshold 1 mm/hr = 1/3600 kg m⁻² s⁻¹)

| quantity | value |
|---|---|
| `prectot_nadir` positive rate | 12.97% (351,615 finite) |
| `prectot_swath` positive rate | 28.93% (351,613 finite) |
| `preccon_nadir` positive rate | 1.03% (351,629 finite) |
| `preccon_swath` positive rate | 4.45% (351,629 finite) |
| swath-positive but nadir-negative, share of swath positives | 55.2% |
| nadir-positive but swath-negative | 0 |
| mean `prectot` nadir / swath | 2.81 × 10⁻⁴ / 1.03 × 10⁻³ kg m⁻² s⁻¹ |
| median-split `fire` threshold | median of `prectot_nadir` ≈ 2.5 × 10⁻⁵ kg m⁻² s⁻¹ (~0.09 mm/hr) |

Reproduce with the snippet in `04_label_finding.md`.

## Temporal structure (`task_outputs/t3_autocorr_report.md`)

| variable | lag-1 ACF | e-folding lag |
|---|---|---|
| `prectot_nadir` | 0.171 | 1 decision |
| `preccon_nadir` | 0.116 | 1 decision |
| `tautot_mean` | 0.628 | 3 decisions |

## Preliminary skill — random 30% split (to be replaced by E1)

| feature / model | ROC-AUC vs. `prectot_nadir` ≥ 1 mm/hr |
|---|---|
| `tautot_mean` | 0.815 |
| `tautot_max` | 0.789 |
| `cldtot_mean` | 0.651 |
| `cldtmp_mean` (inverted) | 0.622 |
| `lwtup_mean` (inverted) | 0.598 |
| `ocean_frac_nadir` | 0.572 |
| logistic regression, 12 features | AUC 0.837, AP 0.368 |

Source: `../trackwise_method.md` §4.3.

## Earlier clustering design (`../clustering_method.md`)

| quantity | value |
|---|---|
| clusters poleward of 60°S | 30.5% of clusters in 5.0% of area (6.1× enrichment) |
| largest merged object | ~9.9 × 10⁵ km² |
| max area over-credit to a grazing pass | 39× |
| `lwtup` → `cldtmp` threshold: high-latitude cells flagged | 53.9% → 7.6% |

## External facts

| fact | value | reference |
|---|---|---|
| learned DT vs. best heuristic | IL +10.0%, RL +13.7% | breitfeld2025learning |
| DT flight | CogniSAT-6, 2025 | dt2025cognisat6 |
| DT lookahead lead time | 60–90 s (early experiments) | ubotica_dt |
| Φ-sat-1 onboard CNN | Myriad 2, launched Sept 2020 | giuffrida2022phisat |
| INCUS | 3 smallsats, RainCube-heritage Ka-band radars, launch 2027 | tanelli2023incus |
| G5NR known bias | split ITCZ, weak Hadley circulation | gelaro2015g5nr |
