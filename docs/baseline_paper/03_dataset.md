# 3. The Track-Wise Dataset

Source documents in this repo: [`../trackwise_method.md`](../trackwise_method.md),
[`../feature_set.md`](../feature_set.md), [`../column_registry.yaml`](../column_registry.yaml),
`task_outputs/final_validation_report.md`.

## 3.1 Design principle

The unit of the dataset is the **decision**, not the storm. An earlier threshold-and-cluster design
(storm objects intersected with satellite passes) produced rows only where a cold region existed,
over-detected polar ice (30.5% of clusters poleward of 60°S, a 6.1× enrichment), and credited a
grazing pass with up to 39× more area than it could image
([`../clustering_method.md`](../clustering_method.md)). The track-wise design produces exactly one
row per minute of flight, whether or not anything is happening below.

## 3.2 Orbit and decision geometry

- Real EarthCARE orbital elements (TLE), sub-satellite point every 1 s.
- Altitude ~394 km; MSI swath 150 km.
- One decision per minute; 60 one-second nadir samples per decision.
- 351,639 decisions; 1,161 missing decision IDs across 49 discontinuities, kept as gaps and never
  interpolated.

## 3.3 Weather fields

- GEOS-5 Nature Run (G5NR), 0.0625° (~7 km), 30-minute output.
- Window: June 2005 – January 2006; 11,731 distinct 30-minute weather states.
- The start month was chosen by an area-weighted global precipitation scan of all 24 G5NR months:
  June–July 2005 is the wettest consecutive pair, so the window opens at peak convective activity
  and then crosses into Northern Hemisphere autumn and winter.
- Weather step is assigned arithmetically (decision → elapsed time → 30-minute index), not by
  nearest timestamp. The orbit clock (TLE epoch, 2025–2026) and the G5NR clock (2005–2006) are
  independent; the start is aligned to EarthCARE's local solar time.

## 3.4 Extraction

For each decision, a small bounding box around the track is fetched from the correct weather step,
reduced to summary values, and discarded. Downloading the full grid instead would need about
10.1 TB (2,881 × 5,760 cells × 13 variables × 11,760 steps × 4 B). Peak memory per request is under
1 MB.

## 3.5 Columns (40 total)

| tier | role | columns |
|---|---|---|
| `always` (12) | policy inputs | `cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`, `lwtup_fore`, `solar_hour`, `lat`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` |
| `cpr_conditional` (5) | revealed only after firing (not used here) | `tauhgh_mean`, `taulow_mean`, `cldhgh_mean`, `cldmid_mean`, `cldlow_mean` |
| `target` (8) | scoring only, never inputs | `prectot`, `preccon`, `iwp`, `lwp`, each `_nadir` and `_swath` |
| `metadata` | IDs, time, raw geometry, counts, validity flags | `decision`, `field`, `g5_step`, `time`, `lon`, `n_swath_cells`, `n_nadir_cells`, 8 × `*_valid` |

- `_swath` = maximum over model cells inside the 150 km MSI swath for that decision.
- `_nadir` = maximum over model cells under the 60 one-second sub-satellite points.
- `lwtup_fore` = next decision's `lwtup_mean`, kept only when the next decision is consecutive and
  in the same weather step. It is a geometric forward-view proxy for BBR, not a BBR retrieval.
- `land_frac_nadir` = `frland + frlandice`; `ocean_frac_nadir` = `frocean + frlake`; partition
  error ≤ 0.00075. Orbit-sampled means: land 0.339, water 0.661.

## 3.6 Proxy framing (state this explicitly in the paper)

The cloud and radiation columns are G5NR model diagnostics named after physical quantities. They are
**not** simulated MSI retrievals: no instrument forward model or retrieval chain was applied.
`tautot` is a column total, while an MSI optical-depth retrieval describes the topmost cloud layer
and needs daylight. Treat all features as idealized proxies; the onboard-feasibility tiers
(Section 6) exist partly to bound how much this matters.

## 3.7 Quality control

- Fill-value sentinels that look like ordinary finite numbers were found in 710 readings (0.20% of
  rows) and set to NaN after physical-range checks on every feature and target.
- Target validation: 94 invalid entries across target columns, in 51 decisions; 24 finite corrupt
  values converted to NaN; legitimate zeros preserved; no imputation.
- Extraction code fixed so the sentinel leak cannot recur.

## 3.8 Temporal structure (matters for the split in Section 7)

Lag-*k* autocorrelation over consecutive decisions only:

| variable | lag 1 | lag 2 | lag 3 | lag 5 | lag 10 | e-folding lag |
|---|---|---|---|---|---|---|
| `prectot_nadir` | 0.171 | 0.060 | 0.016 | 0.011 | 0.006 | 1 min |
| `preccon_nadir` | 0.116 | 0.057 | 0.038 | 0.025 | −0.003 | 1 min |
| `tautot_mean` | 0.628 | 0.370 | 0.230 | 0.067 | −0.054 | 3 min |

Nadir precipitation decorrelates within a minute (the ground track moves ~7 km/s, so ~420 km or
~60 model cells per decision), but consecutive decisions inside the same 30-minute weather step
share one weather field. A random row-level split therefore places near-duplicate weather in train
and test. This is why Section 7 requires a time-blocked split.

## 3.9 Release

- File: `trackwise_decisions.parquet`, 351,639 × 40,
  SHA-256 `4b92f01e24e231a5b5a1151060fef586f08e450e15c252fb23ef9ef772fd1466`.
- Code: `extract_trackwise.py`, `geos5data_0.0625deg_30mn_trackwise.ipynb`.
- Before submission: archive the parquet, code, and registry with a DOI (e.g., Zenodo) and cite it.
- If the dataset is regenerated after the gap-closing retry commit (`4a06337`), update every count
  and the hash in this file and in `baseline_paper.md`.
