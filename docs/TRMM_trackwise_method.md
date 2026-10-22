# The TRMM Track-Wise Method: What Changes from EarthCARE

**Applies to:** the TRMM dataset `missions/trmm/data/TRMM_trackwise_decisions.parquet`, built by
`extract_trackwise.py --mission trmm` and `task_outputs/enrich_trackwise.py --mission trmm`, and
analysed in `missions/trmm/TRMM_trackwise.ipynb`. Settings: `trackwise_missions.py` (`TRMM`) and
`baselines/missions.py` (`TRMM`). Columns: `missions/trmm/column_registry.yaml`.

**Status (Oct 2026):** code complete and checked offline; the extraction has not been run yet.

The method is the one described in [`EarthCare_trackwise_method.md`](./EarthCare_trackwise_method.md):
one row per one-minute decision along the real ground track, a historical TLE set with
closest-epoch propagation, G5NR weather chosen by index arithmetic (step 803 + decision // 30),
a small bounding-box download per three decisions that is reduced and discarded, the same
enrichment, the same time-blocked split and metrics. This document lists only what differs and why.

---

## 1. The mission

TRMM (Tropical Rainfall Measuring Mission, NORAD 25063, 1997–2015) carried the first spaceborne
precipitation radar beside passive sensors on one spacecraft:

| | value | source |
|---|---|---|
| orbit | 34.97° inclination, precessing; samples every local time | TLE set |
| altitude in the window | 398.5–400.6 km (mean altitude above the equatorial radius) | TLE set |
| radar | Precipitation Radar (PR), 13.8 GHz, cross-track scanning, **247 km swath**, 5.0 km resolution | NASA GES DISC (post-boost, Aug 2001 onward) |
| passive imager | Visible and Infrared Scanner (VIRS), 5 channels (0.63, 1.6, 3.75, 10.8, 12 µm), **833 km swath**, 2.4 km resolution | NASA GES DISC (post-boost) |
| microwave imager | TMI (not used: decision D4, VIRS only first) | — |
| broadband radiometer | none in 2005: TRMM's CERES failed in 2000 | NASA LaRC CERES |

Both swaths are centred on the ground track (no roll), unlike EarthCARE's MSI, which is rolled
5.76° to one side. The values are the post-boost ones (TRMM was raised from 350 km to ~400 km in
August 2001); the pre-boost swaths (PR 215 km, VIRS 720 km) do not apply to 2005.

## 2. What changes, and why

| item | EarthCARE | TRMM | why |
|---|---|---|---|
| TLE set | 59908, 593 TLEs | 25063, 380 TLEs, 2005-05-25 → 2006-02-08 | `missions/trmm/tle/sat000025063.txt` |
| largest TLE gap allowed | 3 days | **3.5 days** | the set has one gap of 3 d 0 h 23 min (2005-10-07 → 10); see §3 |
| orbit start | 2025-06-01 15:00 UTC (same day of year as G5NR) | **2005-06-01 15:00 UTC, the same date as G5NR** | TRMM flew during G5NR's own year |
| passive footprint | MSI 150 km, rolled 5.76° | VIRS 833 km, centred | instrument geometry |
| label footprint | CPR nadir line | **PR swath, 247 km, centred** | the PR scans; it measures the whole swath |
| label columns | `*_nadir` (label), `*_swath` | `*_radar` (label), `*_radar_mean`, `*_swath`, `*_nadir` | §4 |
| G5NR variables | 13 | **6**: `tautot, cldtot, cldtmp, lwtup, prectot, preccon` | §5 |
| `lwtup_fore` | yes (BBR fore view) | **no** | VIRS is a cross-track scanner and there was no working BBR-like sensor |
| latitude range | ±82° | ±35.1° nadir, ~±37° PR swath | orbit |
| per-minute ground patch | MSI 150 × ~430 km | VIRS ~840 × ~430 km (features), PR ~249 × ~430 km (label) | measured, §6 |

## 3. Orbit and time alignment

Each orbit minute uses the TLE whose epoch is closest, as for EarthCARE. Because TRMM flew in
2005, the orbit and the weather share the calendar date: decision 0 is 2005-06-01 15:00 UTC for
both, and every decision reads the G5NR snapshot 0–29 min before it.

**The 3-day TLE gap.** CelesTrak's history has no TRMM element sets between 2005-10-07 06:45 and
2005-10-10 07:08 UTC. Near the middle of that gap a moment is up to ~1.5 days from the closest
epoch, so along-track position errors of a few km to a few tens of km are possible. G5NR is a
free-running simulation, not 2005's real weather, so a position error only changes which
simulated weather is sampled; it does not bias any statistic. The limit is therefore set to 3.5
days for TRMM, and the extractor still refuses to run if any gap exceeds it.

## 4. Labels

For each precipitation variable (`prectot`, `preccon`):

| column | definition | role |
|---|---|---|
| `*_radar` | maximum over grid cells inside the PR swath | **label** (rain the PR would measure) |
| `*_radar_mean` | mean over grid cells inside the PR swath | mass-capture weight: for a scanning radar, the rain it collects scales with rain area × rate, not with the maximum |
| `*_swath` | maximum over the VIRS swath | label-geometry comparison (E4) |
| `*_nadir` | maximum over cells under the 60 one-second nadir points | comparison with a nadir-only radar such as EarthCARE's CPR on the same track (E4, E12) |

The threshold stays 1 mm/hr: above the PR's minimum detectable rain rate (~0.5–0.7 mm/hr,
≈ 17 dBZ) and the same as EarthCARE's, so the missions are comparable.

## 5. Features

The VIRS-like features are means over the VIRS swath: `cldtot_mean`, `cldtmp_mean` (stand-in for
the 10.8 µm brightness temperature), `tautot_mean` and `tautot_max` (daytime only), and
`lwtup_mean`. TRMM had no working broadband radiometer in 2005, so `lwtup_mean` is framed as a
proxy for the VIRS thermal-infrared window, not as a BBR measurement. Not extracted for TRMM:

- the cloud-layer diagnostics (`tauhgh`, `taulow`, `cldhgh`, `cldmid`, `cldlow`): for EarthCARE they
  stood for information the CPR reveals after firing; the 13.8 GHz PR does not see cloud layers;
- `iwp`, `lwp`: the PR does not measure cloud ice or liquid water paths;
- `lwtup_fore`: no forward-looking sensor.

This halves the variables per request, which keeps a TRMM download (a ~4× larger box) at about the
same size as an EarthCARE one.

Feature tiers (`baselines/missions.py`): A = the five VIRS-like features + geometry; A−τ = A without
optical depth; B = `cldtmp_mean`, `lwtup_mean` + geometry; C = geometry. Geometry is the same seven
columns as EarthCARE (`lat`, solar-hour sine/cosine, `sin_lon`, `cos_lon`, nadir land/water).
Because TRMM samples every local time, its day/night split has data at every hour and latitude.

## 6. Geometry checks (offline, before extraction)

Run with the repository code on the real TLE set and the real 0.0625° G5NR grid axes (no weather
download), over 240 decisions spread across the window including 12 in dateline tasks:

| check | result |
|---|---|
| TLE coverage | passes: 380 TLEs, largest gap 3 d 0 h 23 min < 3.5 d |
| nadir latitude over all 352,800 minutes | max \|lat\| 35.13° |
| VIRS cross-track width through the nadir point | 827–851 km (mean 840; published 833) |
| PR cross-track width | 245–252 km (mean 249; published 247) |
| PR swath inside the VIRS swath | 99.9% of the PR area (the rest is minute-end corners, still inside the download box) |
| grid cells per decision | VIRS 7,066–8,934; PR 2,085–2,668; never empty |
| largest download box | 46,648 grid cells (× 6 variables ≈ EarthCARE's 18k cells × 13) |

Two shared-code changes came out of these checks and apply to every mission:

- **No geometry fallbacks.** A footprint with no grid-cell centre inside it, or a nadir point whose
  grid cell is not in the downloaded box, now stops the run (`GeometryError`) instead of being
  patched silently. EarthCARE's old fallback had filled 12 decisions at 80.8–82.2°S, where its
  97° orbit turns; those rows remain as extracted (EarthCARE is not re-extracted). TRMM never goes
  poleward of ~37°.
- **Dateline handling.** A task crossing ±180° now downloads two tight boxes, one each side, instead
  of two fixed 60° windows; and each nadir point's grid cell is found with a longitude distance
  that wraps at ±180°, so a point at 179.98° maps to the −180° column.

## 7. Running it

The repository script runs from any working directory; the cache folder is created there:

```
cd ~/trackwise_run && nohup python3 -u "<repo>/extract_trackwise.py" --mission trmm > trmm.log &   # ~30-40 h
python3 task_outputs/enrich_trackwise.py --mission trmm ~/trackwise_run/cache_trmm_8mo_jun2005
python  -m baselines.run_baselines --mission trmm --mlp
python3 task_outputs/build_notebook.py --mission trmm     # then execute missions/trmm/TRMM_trackwise.ipynb
```

E6 uses TRMM's own latitude bands (|lat| < 10°, 10–23.5°, ≥ 23.5°) plus ocean/land/mixed; E4 is a
3 × 3 grid (PR swath, VIRS swath, nadir) for total and convective precipitation.
