# EarthCARE (NORAD 59908)

Sun-synchronous orbit (nodes ~13:56 / 01:56 local), nadir-only Cloud Profiling Radar (CPR), Multi-Spectral Imager (MSI) passive swath. Detection-prioritization benchmark.

| path | contents |
|---|---|
| `tle/sat000059908.txt` | CelesTrak historical TLE set, 2025-05-25 to 2026-02-08 (orbit for the extraction) |
| `data/EarthCare_trackwise_decisions.parquet` | enriched dataset, 352,800 decisions x 40 columns (not in git; rebuild below) |
| `data/enrich_report.json` | row/column counts, sha256, NaN and validity counts for the dataset |
| `results/` | baseline experiments E1-E11: one CSV per table, figures, `run_meta.json` |
| `exports/` | CSV and Cesium CZML exports (not in git) |
| `geos5data_0.0625deg_30mn_trackwise.ipynb` | exploration notebook for this dataset |

Rebuild, from the repository root (the code is shared across missions):

```
python3 extract_trackwise.py                                   # 59 blocks into cache_8mo_jun2005_season/
python3 task_outputs/fetch_g5nr_constants.py                   # once: G5NR land/water grid
python3 task_outputs/enrich_trackwise.py <extraction cache dir>
python  -m baselines.run_baselines --mission earthcare
python3 task_outputs/export_trackwise.py                       # optional: CSV/CZML
```

Method: `docs/trackwise_method.md`. Columns: `docs/column_registry.yaml`, `docs/feature_set.md`.
