# 3. The Track-Wise Datasets

Source documents in this repo: [`../EarthCare_trackwise_method.md`](../EarthCare_trackwise_method.md),
[`../feature_set.md`](../feature_set.md), [`../../missions/earthcare/column_registry.yaml`](../../missions/earthcare/column_registry.yaml),
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

## 3.7 TRMM **[done]**

| item | value |
|---|---|
| NORAD ID | 25063 |
| orbit window | 2005-06-01 15:00 – 2006-02-01 15:00 UTC — the **same date** as G5NR, because TRMM was flying (historical TLE set, 380 TLEs, 2005-05-25 – 2006-02-08; one 3 d 0 h 23 min gap, allowed by a 3.5-day limit) |
| orbit | 34.97°, precessing; samples all local times; 398.5–400.6 km in the window (after the 2001 boost) |
| passive context | VIRS (5 visible/infrared channels, **833 km** swath, centred) → MSI-style cloud proxies over the VIRS swath; `lwtup_mean` stands in for the thermal-infrared window, since TRMM's CERES failed in 2000; no forward-view feature; TMI optional (§3.9) |
| radar footprint | PR, 13.8 GHz, cross-track scanning, **247 km** swath, 5 km resolution → label `prectot_radar` = max over cells in the PR swath; `prectot_radar_mean` (mean over the PR swath) weights mass-capture |
| comparison truth | max over the VIRS swath (`*_swath`) and along the nadir line (`*_nadir`, for comparison with EarthCARE's CPR) |
| G5NR variables | 6 (`tautot, cldtot, cldtmp, lwtup, prectot, preccon`); no cloud-layer diagnostics and no ice/liquid water paths, which the PR does not measure |
| coverage | nadir never beyond 35.1° latitude; the PR swath reaches ~37° |

Sources: NASA GES DISC TRMM product pages (post-boost PR and VIRS figures), NASA LaRC (CERES).
Offline checks with the real TLE set and grid (`../TRMM_trackwise_method.md` §6): measured VIRS width
827–851 km, PR 245–252 km, every footprint covers grid cells.

TRMM reuses the whole EarthCARE pipeline (`extract_trackwise.py --mission trmm`): the passive
features are the same kind of cloud proxies, and the swath widths, the radar footprint and the
variable list change. `solar_hour` now takes all values, so day/night splits (Section 6) have data
on both sides at every latitude.

## 3.8 FY-3G **[code ready; dataset pending]**

| item | value |
|---|---|
| NORAD ID | 56232 |
| orbit window | 2025-06-01 15:00 – 2026-02-01 15:00 UTC, day-of-year matched as for EarthCARE (historical TLE set, 858 element sets, 2025-05-25 – 2026-02-08; largest gap 7 d 23 h across an orbit raise, allowed by an 8-day limit) |
| orbit | 49.99°, precessing; samples all local times; 403–411 km in the window (nominal 407 km) |
| passive context | MERSI-RM (8 visible/infrared channels, **1000 km** swath, centred) → the same cloud proxies as TRMM over the MERSI-RM swath; `lwtup_mean` stands in for the thermal-infrared window (FY-3G has no broadband radiometer); no forward-view feature; MWRI-RM microwave imager not used |
| radar footprint | PMR, Ku 13.6 + Ka 35.5 GHz, cross-track scanning, **303 km** swath, 5 km resolution → label `prectot_radar` = max over cells in the PMR swath; `prectot_radar_mean` weights mass-capture |
| comparison truth | max over the MERSI-RM swath (`*_swath`) and along the nadir line (`*_nadir`) |
| G5NR variables | 6, the same as TRMM |
| coverage | nadir to 50.15° latitude; the MERSI-RM swath reaches ~55° |

Sources: Yang et al. (2025) for the PMR swath, resolution and orbit; the CMA FY-3G presentation at
IPWG-11 (2024) for the MERSI-RM swath; NSMC for the MERSI-RM channels. Offline checks with the real
TLE set and grid (`../FY3G_trackwise_method.md` §7): measured MERSI-RM width 964–1065 km (it widens
with latitude, as a fixed scan angle does over an oblate Earth), PMR 292–321 km, every footprint
covers grid cells.

**Why FY-3G.** MERSI-RM is a visible/infrared imager like VIRS, so the pipeline, features and label
carry over unchanged; G5NR produces no microwave brightness temperatures, so a mission whose only
passive context is microwave would need real data or a forward model. Its 50° orbit also extends
TRMM's all-local-time sampling from ±35° to ±50°, the latitudes where EarthCARE alone samples only
two local times.

## 3.9 Optional: TRMM with TMI

TRMM carried both a visible/infrared scanner and a microwave imager. Running TRMM twice — VIRS only,
then VIRS + TMI — measures directly how much passive microwave information changes gating skill. TMI
measures microwave brightness temperatures, which G5NR does not produce, so this variant needs real
data or a forward model.

## 3.10 Release

- Per mission: one parquet (352,800 rows), its SHA-256, and the TLE set used.
- Code: `extract_trackwise.py` with a per-mission configuration (TLE file, window, swath widths,
  footprint), and the analysis notebook.
- Before submission: archive the parquets, TLE sets, code, and registry with a DOI (e.g., Zenodo) and
  cite it.
