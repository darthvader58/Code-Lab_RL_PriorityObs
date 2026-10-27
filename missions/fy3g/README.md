# FY-3G (NORAD 56232)

50° precessing orbit at ~407 km (all local times), scanning Precipitation Measurement Radar (PMR,
Ku 13.6 / Ka 35.5 GHz, 303 km swath), Medium Resolution Spectral Imager (MERSI-RM, 8 VIS/IR
channels, 1000 km swath). Detection-prioritization benchmark: should the PMR be on this minute,
judged from MERSI-RM-like context? Replaces GPM as the third mission: it samples every local time
to ±50°, beyond TRMM's ±35°, with a VIS/IR imager the G5NR proxies can represent.

**Status:** code complete and checked offline (2026-10-27); extraction not run yet.
Method and differences from EarthCARE and TRMM: `docs/FY3G_trackwise_method.md`.

| path | contents |
|---|---|
| `tle/sat000056232.txt` | CelesTrak historical TLE set, 2025-05-25 to 2026-02-08 (858 element sets, 823 unique epochs; largest gap 7 d 23 h 37 min across an orbit raise, allowed by the 8 d limit) |
| `column_registry.yaml` | the 35 dataset columns and their tiers |
| `data/FY3G_trackwise_decisions.parquet` | enriched dataset, 352,800 decisions x 35 columns (after the run; not in git) |
| `data/enrich_report.json` | row/column counts, sha256, NaN and validity counts (after the run) |
| `results/` | baseline experiments E1-E11 (after the run) |
| `exports/`, `cesium/` | CSV and Cesium CZML exports (after the run) |
| `FY3G_trackwise.ipynb` | analysis notebook, built but not executed until the dataset exists |

Run, from any working directory (the cache folder is created there). The extraction needs the
interpreter that has TATC 3.4.10 (the `trackwise-py` kernel's Python,
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`); Homebrew's `python3` does not:

```
cd ~/trackwise_run && nohup caffeinate -is /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    -u "<repo>/extract_trackwise.py" --mission fy3g > fy3g.log 2>&1 &                             # ~35-50 h
python3 task_outputs/enrich_trackwise.py --mission fy3g ~/trackwise_run/cache_fy3g_8mo_jun2005_season
python  -m baselines.run_baselines --mission fy3g --mlp
python3 task_outputs/export_trackwise.py --mission fy3g                                           # optional
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=trackwise-py \
    missions/fy3g/FY3G_trackwise.ipynb
```
