# 3. The Track-Wise Datasets

Source documents in this repo: [`../EarthCare_trackwise_method.md`](../EarthCare_trackwise_method.md),
[`../feature_set.md`](../feature_set.md), [`../column_registry.yaml`](../column_registry.yaml),
`extract_trackwise.py`, `task_outputs/enrich_trackwise.py`, and per mission `missions/<mission>/`.

One dataset is built per mission with the same method (§3.1–3.5). The missions differ in orbit,
passive sensor, and radar footprint (§3.6–3.8). Each dataset is trained and validated on its own.

## 3.1 Design principle

The unit of each dataset is the **decision**, not the storm. An earlier threshold-and-cluster design
(storm objects intersected with satellite passes) produced rows only where a cold region existed,
over-detected polar ice (30.5% of clusters poleward of 60°S, a 6.1× enrichment), and credited a
grazing pass with up to 39× more area than it could image
([`../clustering_method.md`](../clustering_method.md)). The track-wise design produces exactly one
row per minute of flight, whether or not anything is happening below.

## 3.2 Orbit and decision geometry (all missions)

- **Historical orbital elements.** Each mission's orbit comes from a set of historical TLEs requested
  from CelesTrak for the study window, not a single TLE. A set captures the orbit maintenance flown
  during the window (altitude corrections, small adjustments). For each moment, TATC footprints and
  the per-second nadir points both use the TLE whose epoch is closest, so footprints and labels come
  from the same propagation. TATC's repeat-cycle shortcut is switched off.
- One decision per minute; 60 one-second sub-satellite samples per decision.
- 352,800 decisions per mission (11,760 G5NR steps × 30 decisions). The extractor retries failed
  downloads and saves a block only when complete, so the datasets have no gaps by construction.

## 3.3 Weather fields and time alignment (all missions)

- GEOS-5 Nature Run (G5NR), 0.0625° (~7 km), 30-minute output.
- Window: 2005-06-01 15:00 – 2006-02-01 15:00 UTC (G5NR steps 803–12562). June–July 2005 is the
  wettest consecutive pair of the 24 G5NR months (area-weighted global precipitation scan), so the
  window opens at peak convective activity and then crosses into Northern Hemisphere autumn and
  winter.
- **Time alignment.** The orbit window and the weather window start at the same **UTC time of day**
  (15:00) and the same **day of year** (1 June), so each decision reads the latest G5NR snapshot at
  or before the satellite's time (offset 0–29 min) and the weather comes from the same season as the
  orbit geometry. Neither 8-month span contains a 29 February, so the day of year stays aligned.
  The calendar year can differ; only time of day and season must match.
- The weather step is assigned arithmetically (decision → elapsed time → 30-minute index), never by
  matching timestamps across the two clocks.

## 3.4 Extraction (all missions)

For each decision, a small bounding box around the track is fetched from the correct weather step,
reduced to summary values, and discarded. Downloading the full grid instead would need about
10.1 TB (2,881 × 5,760 cells × 13 variables × 11,760 steps × 4 B). Peak memory per request is under
1 MB.

## 3.5 Columns and proxy framing (all missions)

| tier | role | columns (EarthCARE; other missions in §3.7–3.8) |
|---|---|---|
| `always` | policy inputs | `cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`, `lwtup_fore`, `solar_hour`, `lat`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` |
| `cpr_conditional` | revealed only after firing (not used here) | `tauhgh_mean`, `taulow_mean`, `cldhgh_mean`, `cldmid_mean`, `cldlow_mean` |
| `target` | scoring only, never inputs | `prectot`, `preccon`, `iwp`, `lwp`, each over the radar footprint and over the passive swath |
| `metadata` | IDs, time, geometry, counts, validity flags | `decision`, `field`, `g5_step`, `time`, `lon`, cell counts, `*_valid` |

The cloud and radiation columns are G5NR model diagnostics named after physical quantities. They are
**not** simulated instrument retrievals: no forward model or retrieval chain was applied. `tautot` is
a column total, while an imager optical-depth retrieval describes the topmost cloud layer and needs
daylight. Treat all features as idealized proxies; the onboard-feasibility tiers (Section 6) exist
partly to bound how much this matters.

## 3.6 EarthCARE (primary case)

| item | value |
|---|---|
| NORAD ID | 59908 |
| orbit window | 2025-06-01 15:00 – 2026-02-01 15:00 UTC (historical TLE set, 2025-05-25 – 2026-02-08) |
| orbit | sun-synchronous, ~97°; nodes at 13:56–13:57 (descending) and 01:56–01:57 (ascending) local time over the window |
| passive context | MSI, 150 km swath → `*_mean`, `tautot_max` over the swath |
| radar footprint | CPR, nadir only → `*_nadir` = max over cells under the 60 one-second nadir points |
| comparison truth | `*_swath` = max over cells in the 150 km MSI swath |
| dataset | `missions/earthcare/data/EarthCare_trackwise_decisions.parquet`: 352,800 × 40, decisions 0–352,799, no gaps, SHA-256 `9fe09a93…2424` |
| status | **complete** (extraction 30.6 h; enrichment; E1–E11) |

Quality control and structure:

- **Fill values** (finite numbers like 10³⁵ that G5NR returns for missing data) are dropped at
  extraction, before averaging.
- **One corrupted server response** put physically impossible precipitation (up to 6 × 10⁴
  kg m⁻² s⁻¹) into 3 decisions; the source grid was normal, so that block of 6,000 decisions was
  re-downloaded, and the re-download matched the original everywhere else. A physical-range audit of
  every weather column then finds **no out-of-range value** in the 352,800 rows.
- **Validity.** All 8 targets are finite and valid in every row. The only missing features are
  `cldtmp_mean` (4.9%; no cloud top anywhere in the swath, none rainy) and `lwtup_fore` (one row per
  weather step, at the step boundary).
- **Lag-*k* autocorrelation** over consecutive decisions: `prectot_nadir` lag-1 0.106 (e-folding
  1 min); `preccon_nadir` 0.114 (1 min); `tautot_mean` 0.639 (3 min); `lwtup_mean` 0.838 (6 min).
- **Orbit-sampled surface** at nadir: land 0.339, water 0.661; 17.1% of decisions mixed.
- **Local-time sampling:** 82% of minutes fall within the six local hours around 14:00 and 02:00;
  other hours are reached only near the poles.

Nadir precipitation decorrelates within a minute (the ground track moves ~7.5 km/s, so ~450 km or ~60
model cells per decision), but consecutive decisions inside the same 30-minute weather step share
one weather field. A random row-level split therefore places near-duplicate weather in train and
test. This holds for every mission and is why Section 7 requires a time-blocked split. (For
EarthCARE the effect on logistic regression turned out small: AUC 0.833 random vs. 0.837
time-blocked; E1.)

## 3.7 TRMM **[TO BUILD]**

| item | value |
|---|---|
| NORAD ID | 25063 |
| orbit window | 2005-06-01 15:00 – 2006-02-01 15:00 UTC — the **same year** as G5NR, because TRMM was flying (historical TLE set, 2005-05-25 – 2006-02-08) |
| orbit | 35°, precessing (~46-day cycle relative to the Sun); samples all local times; ~402 km after the 2001 boost |
| passive context | VIRS (5 visible/infrared channels) → same MSI-style cloud proxies over the VIRS swath (~720 km); TMI optional (§3.9) |
| radar footprint | PR, 13.8 GHz, cross-track scanning, ~245 km swath → label = max over cells in the PR swath |
| comparison truth | max over cells in the VIRS swath |
| coverage | the track never goes beyond ~35° latitude; the PR swath reaches a few degrees further |

TRMM reuses almost the whole EarthCARE pipeline: the passive features are the same kind of cloud
proxies, and only the swath widths and the radar footprint change. `solar_hour` now takes all values,
so day/night splits (Section 6) have data on both sides at every latitude.

## 3.8 GPM Core Observatory **[TO BUILD; data source open]**

| item | value |
|---|---|
| NORAD ID | 39574 |
| orbit window | 2025-06-01 15:00 – 2026-02-01 15:00 UTC, day-of-year matched (historical TLE set, 2025-05-25 – 2026-02-08) |
| orbit | 65°, precessing, ~407 km; samples all local times |
| passive context | GMI, 13 channels 10–183 GHz, conical scan, ~885 km swath, footprint ahead of the spacecraft |
| radar footprint | DPR (KuPR 13.6 GHz + KaPR 35.5 GHz), cross-track scanning, ~245 km swath |
| comparison truth | max over cells in the GMI swath |

The GPM case cannot simply reuse the cloud proxies, because GMI measures microwave brightness
temperatures, which respond directly to rain (emission, low frequencies) and ice (scattering, high
frequencies). G5NR does not produce brightness temperatures. Three options, decided before building
(`14_submission_plan.md`, decision D2):

1. **Real GPM data.** GMI brightness temperatures and DPR products are public and already collocated
   on one spacecraft. No simulation; the label comes from DPR itself. Preferred.
2. **Microwave forward model on G5NR** (e.g., RTTOV or CRTM) to simulate GMI brightness temperatures.
   Consistent with the other two cases, but substantial work.
3. **G5NR proxies** (column water vapour, liquid and ice water paths). Cheapest, but these are closer
   to what the radar measures than to what GMI sees, so they make the task too easy; usable only as a
   clearly labelled upper bound.

The GPM label is also different: see `05_baselines.md` §5.8.

## 3.9 Optional: TRMM with TMI

TRMM carried both a visible/infrared scanner and a microwave imager. Running TRMM twice — VIRS only,
then VIRS + TMI — measures directly how much passive microwave information changes gating skill. TMI
faces the same brightness-temperature problem as GMI (§3.8), so this variant inherits decision D2.

## 3.10 Release

- Per mission: one parquet (352,800 rows), its SHA-256, and the TLE set used.
- Code: `extract_trackwise.py` with a per-mission configuration (TLE file, window, swath widths,
  footprint), and the analysis notebook.
- Before submission: archive the parquets, TLE sets, code, and registry with a DOI (e.g., Zenodo) and
  cite it.
