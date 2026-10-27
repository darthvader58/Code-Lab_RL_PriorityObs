# The FY-3G Track-Wise Method: What Changes from EarthCARE and TRMM

**Applies to:** the FY-3G dataset `missions/fy3g/data/FY3G_trackwise_decisions.parquet`, built by
`extract_trackwise.py --mission fy3g` and `task_outputs/enrich_trackwise.py --mission fy3g`, and
analysed in `missions/fy3g/FY3G_trackwise.ipynb`. Settings: `trackwise_missions.py` (`FY3G`) and
`baselines/missions.py` (`FY3G`). Columns: `missions/fy3g/column_registry.yaml`.

**Status (Oct 2026):** code complete and checked offline; the extraction has not been run yet.

The method is the one described in [`EarthCare_trackwise_method.md`](./EarthCare_trackwise_method.md),
with the scanning-radar labels introduced for TRMM in [`TRMM_trackwise_method.md`](./TRMM_trackwise_method.md).
This document lists only what differs and why.

---

## 1. Why FY-3G

FY-3G carries a visible/infrared imager next to its radar, so the G5NR cloud proxies and the "rain
under the radar" label carry over unchanged, as they did from EarthCARE to TRMM. G5NR has no
microwave brightness temperatures, so the microwave imager (MWRI-RM) is not used.

It also fills the gap the first two missions leave together. EarthCARE samples all latitudes to
±82° but only near 01:56 and 13:56 local time; TRMM samples every local time but only to ±35°.
FY-3G's 50° precessing orbit samples every local time to ±50°, so it adds the subtropics and
lower mid-latitudes (35–50°) at all hours, where frontal precipitation dominates.

## 2. The mission

FY-3G (Fengyun-3G, NORAD 56232, COSPAR 2023-055A, launched 2023-04-16) is China's first
precipitation-measurement satellite:

| | value | source |
|---|---|---|
| orbit | 49.99° inclination, precessing; samples every local time | TLE set |
| altitude in the window | 403–411 km (mean altitude above the equatorial radius); nominal 407 km | TLE set; Yang et al. (2025) |
| radar | Precipitation Measurement Radar (PMR), Ku 13.6 GHz and Ka 35.5 GHz, cross-track scanning, **303 km swath**, 5 km resolution at nadir | Yang et al. (2025); CMA (IPWG-11, 2024) |
| passive imager | Medium Resolution Spectral Imager (MERSI-RM), 8 channels (0.65, 0.865, 0.94, 1.38, 1.64, 3.8, 10.8, 12.0 µm), **1000 km swath**, 500 m resolution | NSMC instrument page; CMA (IPWG-11, 2024) |
| microwave imager | MWRI-RM (not used, as TMI was not for TRMM) | — |
| broadband radiometer | none | — |

Both swaths are taken as centred on the ground track. The PMR swath is the one published value
(303 km); the Ka beam is treated as matched to Ku. Using the Ku swath
only would not change the label geometry.

## 3. What changes, and why

| item | EarthCARE | TRMM | FY-3G | why |
|---|---|---|---|---|
| TLE set | 59908, 593 TLEs | 25063, 380 TLEs | 56232, **823 TLEs**, 2025-05-25 → 2026-02-08 | `missions/fy3g/tle/sat000056232.txt` (858 element sets, 823 unique epochs) |
| largest TLE gap allowed | 3 days | 3.5 days | **8 days** | the set has a 7 d 23 h 37 min gap; see §4 |
| orbit start | 2025-06-01 15:00 UTC | 2005-06-01 15:00 UTC | **2025-06-01 15:00 UTC** | FY-3G flies in 2025; aligned by day of year, as EarthCARE |
| passive footprint | MSI 150 km, rolled 5.76° | VIRS 833 km | **MERSI-RM 1000 km**, centred | instrument geometry |
| label footprint | CPR nadir line | PR swath, 247 km | **PMR swath, 303 km**, centred | the PMR scans; it measures the whole swath |
| label columns | `*_nadir`, `*_swath` | `*_radar`, `*_radar_mean`, `*_swath`, `*_nadir` | same as TRMM | §5 |
| G5NR variables | 13 | 6 | **6**, same as TRMM | §6 |
| `lwtup_fore` | yes | no | **no** | MERSI-RM is a cross-track scanner; no forward-looking sensor |
| latitude range | ±82° | ±35.1° | **±50.2° nadir**, ~±55° MERSI-RM swath | orbit |
| E6 latitude bands | 4 bands to polar | < 10°, 10–23.5°, ≥ 23.5° | **< 23.5°, 23.5–35°, ≥ 35°** | the ≥ 35° band is what FY-3G adds over TRMM |

## 4. Orbit and time alignment

Each orbit minute uses the TLE whose epoch is closest. FY-3G flies in 2025–26 and G5NR covers
2005–06, so, as for EarthCARE, each orbit minute is paired with the G5NR snapshot at the same day
of year and UTC time (orbit 2025-06-01 15:00 ↔ G5NR 2005-06-01 15:00, step 803). Neither span
contains a 29 February, so the pairing never slips. Local solar time comes from the orbit's UTC
time and longitude, which are the weather's too, so the diurnal cycle is sampled correctly.

**TLE gaps and orbit raises.** The set has five gaps longer than two days. Three are quiet: the
element sets on either side agree to ~3 km at the gap's midpoint. Two contain an orbit raise:

| gap | length | altitude before → after | position difference at the midpoint |
|---|---|---|---|
| 2025-08-05 → 08-08 | 3 d 13 h | 403.1 → 411.1 km | ~2,160 km (along-track) |
| 2025-09-16 → 09-24 | 7 d 23 h | 403.9 → 410.3 km | ~4,490 km (along-track) |

Closest-epoch propagation switches TLE at the gap's midpoint, so the ground track jumps along the
orbit once in each of these gaps: the minutes before the switch follow the pre-raise orbit, those
after it the post-raise orbit. Inclination (49.99°) and the plane's precession are the same on
both sides, so latitude and local-time sampling are unchanged. As for TRMM's gap, G5NR is a
free-running simulation, so a position error changes which simulated weather is sampled, not any
statistic. The limit is 8 days, and the extractor still refuses to run if any gap exceeds it.

## 5. Labels

As for TRMM, for each precipitation variable (`prectot`, `preccon`):

| column | definition | role |
|---|---|---|
| `*_radar` | maximum over grid cells inside the PMR swath | **label** (rain the PMR would measure) |
| `*_radar_mean` | mean over grid cells inside the PMR swath | mass-capture weight (rain area × rate) |
| `*_swath` | maximum over the MERSI-RM swath | label-geometry comparison (E4) |
| `*_nadir` | maximum over cells under the 60 one-second nadir points | comparison with a nadir-only radar such as EarthCARE's CPR (E4, E12) |

The threshold stays 1 mm/hr, the same as EarthCARE's and TRMM's, so the three missions are
comparable. The PMR's Ka band detects lighter rain than TRMM's PR; a lower threshold is an E7
sensitivity, not the primary label.

## 6. Features

The same six G5NR variables as TRMM. The MERSI-RM-like features are means over the MERSI-RM
swath: `cldtot_mean`, `cldtmp_mean` (stand-in for the 10.8 µm brightness temperature),
`tautot_mean` and `tautot_max` (daytime only), and `lwtup_mean`. FY-3G carries no broadband
radiometer, so `lwtup_mean` is a proxy for the MERSI-RM thermal-infrared window (10.8/12 µm).
The cloud-layer diagnostics, `iwp`, `lwp` and `lwtup_fore` are left out for the reasons given for
TRMM. MERSI-RM's 1.38 µm cirrus channel could justify a high-cloud feature (`tauhgh`); it is left
out so that FY-3G's feature set equals TRMM's and E12 compares orbits, not inputs.

Feature tiers (`baselines/missions.py`) are TRMM's: A = the five imager-like features + geometry;
A−τ = A without optical depth; B = `cldtmp_mean`, `lwtup_mean` + geometry; C = geometry.

## 7. Geometry checks (offline, before extraction)

Run with the repository code on the real TLE set and the real 0.0625° G5NR grid axes (no weather
download), over 240 decisions in tasks of three, spread across the window, including 12 in
dateline tasks. The same script reproduces TRMM's checks in `TRMM_trackwise_method.md` §6.

| check | result |
|---|---|
| TLE coverage | passes: 823 TLEs, largest gap 7 d 23 h 37 min < 8 d |
| nadir latitude over all 352,800 minutes | max \|lat\| 50.15° |
| local-time sampling | every 3-hour bin within ±1.5% of an equal share |
| MERSI-RM cross-track width through the nadir point | 964–1065 km (mean 1015; published 1000) |
| PMR cross-track width | 292–321 km (mean 307; published 303) |
| PMR swath inside the MERSI-RM swath | 99.8% of the PMR area (the rest is minute-end corners, inside the download box) |
| grid cells per decision | MERSI-RM 8,433–14,196; PMR 2,555–4,280; never empty |
| largest download box | 71,581 grid cells (× 6 variables; TRMM 46,429) |

The swath widens with latitude (MERSI-RM 1001 km at |lat| < 20°, 1029 km at 40–50°): the orbit is
nearly circular but the Earth is oblate, so the surface lies further below the satellite at higher
latitude and a fixed scan angle covers more ground, as the real instrument does. Altitude changes
in the window barely matter (correlation 0.11 with width).

The download boxes are ~1.5× TRMM's, from the wider swath and from longitude cells narrowing
towards 50°. Expect the extraction to take somewhat longer than TRMM's.

## 8. Running it

```
cd ~/trackwise_run && nohup caffeinate -is /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    -u "<repo>/extract_trackwise.py" --mission fy3g > fy3g.log 2>&1 &
python3 task_outputs/enrich_trackwise.py --mission fy3g ~/trackwise_run/cache_fy3g_8mo_jun2005_season
python  -m baselines.run_baselines --mission fy3g --mlp
python3 task_outputs/build_notebook.py --mission fy3g     # then execute missions/fy3g/FY3G_trackwise.ipynb
```

E4 is a 3 × 3 grid (PMR swath, MERSI-RM swath, nadir) for total and convective precipitation.

## References

- Yang, Y., Han, W., Sun, H., Li, J., Yan, J., and Gao, Z. (2025). Reconstruction of 3D
  precipitation measurements from FY-3G MWRI-RM imaging and sounding channels. *Atmospheric
  Measurement Techniques*, 18, 4249–4269.
- Chen, L. (2024). FY-3G precipitation measurement mission. 11th IPWG Workshop, Tokyo.
  https://www.eorc.jaxa.jp/IPWG/meetings/tokyo-2024/pdf/Session03/3.3_IPWG-11_Chen_Lin_2.pdf
- NSMC. Medium Resolution Spectral Imager-RM (MERSI-RM).
  https://www.nsmc.org.cn/nsmc/en/instrument/MERSI-RM.html
