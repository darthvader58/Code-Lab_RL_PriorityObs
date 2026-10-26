# 13. Key Numbers and Where They Come From

Every number used in the paper, with its source. Update this file whenever a dataset is
regenerated.

**Status (18 Oct 2026):** EarthCARE is complete. Every EarthCARE number below comes from the
season-aligned extraction, dataset SHA-256 `9fe09a93c8b0640f73f23d8422f869573ca172c90c242a4536fdd08a3eb82424`
(`missions/earthcare/data/enrich_report.json`), and the baseline run in
`missions/earthcare/results/` (`run_meta.json` records the same hash). The earlier extraction
(351,639 rows, time-shifted weather) is deleted and none of its numbers are used. TRMM numbers are
in the cross-mission section (E12); GPM has no numbers yet.

## Missions (fixed facts)

| quantity | EarthCARE | TRMM | GPM Core | source |
|---|---|---|---|---|
| NORAD ID | 59908 | 25063 | 39574 | celestrak_satcat (checked Oct 2026) |
| international ID | 2024-101A | 1997-074A | 2014-009C | same |
| launch / end | 2024-05-28 / operating | 1997-11-27 / re-entered 2015-06-16 | 2014-02-27 / operating | same |
| inclination | ~97° (sun-synchronous) | ~35° | ~65° | same |
| orbit window used | 2025-06-01 → 2026-02-01 15:00 UTC | 2005-06-01 → 2006-02-01 15:00 UTC | 2025-06-01 → 2026-02-01 15:00 UTC | `03_dataset.md` |
| TLE set requested | 2025-05-25 → 2026-02-08 | 2005-05-25 → 2006-02-08 | 2025-05-25 → 2026-02-08 | CelesTrak special request |
| local-time nodes | 13:56–13:57 / 01:56–01:57 | all local times | all local times | equator crossings of the nadir track over the window (EarthCARE) |
| radar swath | nadir only | 247 km (PR, post-boost) | ~245 km | trmm_instruments (verified), gpm_instruments (verify) |
| passive swath | MSI 150 km | VIRS 833 km | GMI ~885 km | trmm_instruments (verified), gpm_instruments (verify) |
| altitude in window | ~394 km | 398.5–400.6 km | ~407 km | TLE sets (EarthCARE, TRMM); gpm_instruments |
| TLEs / largest gap | 593 / 2 d 13.7 h | 380 / 3 d 0 h 23 min | — | TLE sets |
| decisions per dataset | 352,800 | 352,800 | 352,800 | 11,760 G5NR steps × 30 |

## EarthCARE dataset

| quantity | value | source |
|---|---|---|
| decisions (rows) × columns | 352,800 × 40, decisions 0–352,799, no gaps | `enrich_report.json` |
| period (G5NR time) | 2005-06-01 15:00 – 2006-02-01 15:00 UTC (steps 803–12,562) | same |
| orbit time (`time` column) | 2025-06-01 15:00 – 2026-02-01 14:59 UTC | parquet |
| distinct 30-min weather states | 11,760 | parquet |
| weather offset from satellite time | 0–29 min (mean 14.5) | 803 + decision // 30 |
| TLE set | 593 TLEs, epochs 2025-05-25 – 2026-02-08, largest gap 2 d 13.7 h | `missions/earthcare/tle/sat000059908.txt` |
| model resolution | 0.0625° (~7 km), 30 min | G5NR |
| MSI roll / swath | 5.76° / 150 km | `extract_trackwise.py` |
| footprint centroid to nadir | 36–49 km (1st–99th pct., median 41 km) | recomputed from the TLE-set orbit |
| locations replaced by nadir point (> 100 km off) | 5,818 (1.6%) | `enrich_report.json` |
| nadir samples per decision | 60 (1 s) | `extract_trackwise.py` |
| full-download size avoided | ~10.1 TB (9.2 TiB) | `../EarthCare_trackwise_method.md` §3 |
| extraction time / server errors absorbed | 30.6 h, 59 blocks / 3,071 (all recovered) | extraction log; method §7.6 |
| corrupted server response | 3 decisions (block 13), re-downloaded | method §4.5 |
| physical-range audit | 0 out-of-range values in 19 weather columns | notebook §2 |
| invalid target entries | 0 of 352,800 × 8 | `enrich_report.json` |
| `cldtmp_mean` NaN | 17,425 (4.9%; no cloud top in swath, none rainy) | same |
| `lwtup_fore` finite / NaN | 341,040 / 11,760 (one per weather step) | same |
| orbit-sampled land / water mean | 0.339 / 0.661 | same |
| mixed-surface decisions (0.05–0.95 land) | 17.11% | same |
| minutes within 6 h around 14:00/02:00 local | 82% | notebook §4 |
| CPR / ATLID power | ~308 W / ~310 W | `../feature_set.md` |

## EarthCARE labels (threshold 1 mm/hr = 1/3600 kg m⁻² s⁻¹; all rows finite)

| quantity | value |
|---|---|
| `prectot_nadir` positive rate | 13.65% (whole dataset); 12.76% in the Nov–Jan test block |
| `prectot_swath` positive rate | 29.99% (whole dataset); 27.85% in the test block |
| `preccon_nadir` positive rate | 1.16% |
| `preccon_swath` positive rate | 5.04% |
| swath-label inflation, `prectot` / `preccon` | 2.2× / 4.3× |
| swath-positive but nadir-negative, share of swath positives | 54.5% (54.2% in the test block) |
| nadir-positive but swath-negative | 0 |
| mean `prectot` nadir / swath | 3.01 × 10⁻⁴ / 1.15 × 10⁻³ kg m⁻² s⁻¹ (3.8×) |
| max `prectot` nadir / swath | 247 / 316 mm/hr |
| median-split `fire` threshold | median of `prectot_nadir` = 2.68 × 10⁻⁵ kg m⁻² s⁻¹ (0.097 mm/hr) |

Reproduce with the snippet in `04_label_finding.md`.

## EarthCARE temporal structure

| variable | lag-1 ACF | e-folding lag |
|---|---|---|
| `prectot_nadir` | 0.106 | 1 decision |
| `preccon_nadir` | 0.114 | 1 decision |
| `tautot_mean` | 0.639 | 3 decisions |
| `cldtot_mean` | 0.651 | 3 decisions |
| `lwtup_mean` | 0.838 | 6 decisions |

## EarthCARE baseline results (time-blocked test, Nov 2005 – Jan 2006; `missions/earthcare/results/`)

Split: train Jun–Sep 2005, validation Oct 2005, test Nov 2005 – Jan 2006 (G5NR time), one weather
step dropped at each boundary. Train / val / test rows: 174,750 / 44,580 / 133,350 (120 dropped).
Tier A = 13 inputs (`cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`,
`lwtup_fore`, `lat`, `sin_sh`, `cos_sh`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir`).

| policy (E3) | AUC [95% CI] | AP | capture@13% | capture@20% | mass-capture@20% |
|---|---|---|---|---|---|
| P1 random | 0.501 | 0.127 | 12.9% | 19.7% | 20.2% |
| P2 geometry LR | 0.593 [0.589, 0.598] | 0.166 | 18.2% | 27.5% | 31.8% |
| P3 latitude band | 0.629 | 0.179 | 18.5% | 28.0% | 34.5% |
| P4 single threshold | 0.718 [0.715, 0.722] | 0.263 | 29.0% | 41.8% | 53.1% |
| P5 logistic regression | 0.837 [0.834, 0.840] | 0.408 | 45.5% | 60.3% | 71.7% |
| P6 LightGBM | **0.893 [0.891, 0.896]** | **0.544** | **50.8%** | **66.4%** | **72.7%** |
| P7 MLP 64-64 | 0.889 [0.887, 0.891] | 0.527 | 50.4% | 65.1% | 71.8% |
| P8 oracle | 1.000 | 1.000 | 100% | 100% | 91.7% |

| quantity | value | experiment |
|---|---|---|
| P6 lift at 20% duty | 3.4× random's capture (66.4% / 19.7%); precision ÷ base rate (E3 `lift` column) 3.5× | E3 |
| P6 − P7 paired difference | AUC +0.004 [+0.003, +0.005]; capture@20% +1.4 pts [+0.8, +2.0]; capture@13% +0.4 pts [−0.1, +0.9] | E10 |
| P6 − P5 paired difference | AUC +0.056 [+0.054, +0.059]; capture@20% +6.1 pts [+5.3, +7.0] | E10 |
| random vs. time-blocked split (P5, d = 13%) | AUC 0.833 vs. 0.837; capture 43.9% vs. 45.5% | E1 |
| leave-one-month-out AUC (P5 / P6 / P7) | 0.811 – 0.844 / 0.887 – 0.907 / 0.879 – 0.905 over 8 months | E2 |
| Nov–Jan AUC, trained Jun–Sep vs. on the other 7 months (P5 / P6 / P7) | 0.837 → 0.843 / 0.893 → 0.905 / 0.889 → 0.903 (gain +0.006 / +0.012 / +0.014; upper bound on the seasonal penalty) | E2, E3 |
| nadir-trained policy, capture@20% on nadir / swath labels | 66.4% / 51.1% | E4 |
| swath-trained policy, capture@20% on nadir / swath labels | 64.7% / 50.7% | E4 |
| precision@20%, nadir-trained, on nadir / swath labels (`prectot`) | 44.7% / 75.0% | E4 |
| convective: test positive rate nadir / swath; swath-positive & nadir-negative | 1.2% / 5.2%; 77.3% | E4 |
| convective: precision@20% on nadir / swath labels; AUC range | 5.9% / 25.5%; 0.957–0.965 | E4 |
| day capture@20%, Tier A / A without `tautot` / B / C | 67.6% / 55.1% / 55.9% / 33.2% | E5 |
| night capture@20%, A without `tautot` / B / C | 49.8% / 49.8% / 33.1% | E5 |
| switched day/night retention vs. Tier A | A−τ 89.2%, B 90.2%, C 56.7% | E5 |
| AUC by region | tropics 0.846, subtropics 0.876, mid-latitudes 0.874, polar 0.940, ocean 0.861, land 0.950, mixed 0.912 | E6 |
| duty used by region at the global 20% threshold | tropics 30.2%, polar 6.2% | E6 |
| P6 AUC by label | ≥ 0.1 mm/hr 0.926, ≥ 1 0.893, ≥ 5 0.922, convective ≥ 1 0.957, median split 0.926 | E7 |
| capture at d = base rate, ≥ 5 mm/hr / convective | 41.1% / 30.4% (P6) | E7 |
| policy size | P5 16 params, 3.3 kB; P7 5,249 params, 134 kB; P6 18,300 params (300 trees), 1.05 MB | E9 |
| wall time per decision (Python, laptop CPU) | 0.8–1.3 ms for all learned policies (varies ±0.3 ms between runs) | E9 |
| top permutation importance (AUC drop) | `tautot_max` 0.207, `tautot_mean` 0.091, `lwtup_mean` 0.052, `cldtot_mean` 0.038 | E11 |
| best single-feature AUC | `tautot_mean` 0.834, `tautot_max` 0.812, `cldtot_mean` 0.652, `cldtmp_mean` 0.619 (inv.) | E11 |

## Cross-mission comparison, EarthCARE vs. TRMM (`missions/comparison/results/`)

TRMM dataset SHA-256 `fab9c8c7eb…7e9c01` (`missions/trmm/data/enrich_report.json`). Training and
testing restricted to |lat| ≤ 35.21° (TRMM's highest latitude): 138,424 EarthCARE and 352,800 TRMM
decisions. EarthCARE windows = ±1 h of 01:56 and 13:56 local time.

| quantity | value | source |
|---|---|---|
| base rate, own radar label | EarthCARE 17.7% (nadir), TRMM 41.1% (PR swath) | E12 comparison |
| swath / radar-label positive rate | EarthCARE 2.23×, TRMM 1.54× | E12 comparison |
| P6 AUC / capture@20% | EarthCARE 0.855 / 61.2%; TRMM 0.849 / 40.7% (oracle 48.7%) | E12 comparison |
| Tier B retention at 20% (P6) | EarthCARE 78.5%, TRMM 79.3% | E12 comparison |
| transfer loss, capture@20%, EarthCARE → TRMM | P6 1.2 points, P5 1.9 (nadir label, shared features); largest in either direction 2.5 | E12 transfer |
| TRMM capture@20% in EarthCARE windows vs. other hours | P6 43.9% vs. 40.1%, gap 3.8 [0.8, 6.8] points (radar label); 3.0–3.9 for every model and label | E12 local time |
| EarthCARE windows' share of local hours / of TRMM rain amount | 16.7% / 17.2% (land 17.3%, ocean 17.1%) | E12 diurnal |
| TRMM rain-amount peak, local time | land 16:00, ocean 03:00 | E12 diurnal |

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
