# TRMM (NORAD 25063)

35° precessing orbit at ~400 km (all local times), scanning Precipitation Radar (PR, 13.8 GHz,
247 km swath), Visible and Infrared Scanner (VIRS, 833 km swath). Detection-prioritization
benchmark: should the PR be on this minute, judged from VIRS-like context?

**Status:** extracted (2026-10-26), enriched, baselines E1-E11 with P7 run, notebook executed; E12 (cross-mission comparison) not yet built.
Method and differences from EarthCARE: `docs/TRMM_trackwise_method.md`.

| path | contents |
|---|---|
| `tle/sat000025063.txt` | CelesTrak historical TLE set, 2005-05-25 to 2006-02-08 (380 TLEs; one 3 d 0 h 23 min gap, allowed by the 3.5 d limit) |
| `column_registry.yaml` | the 35 dataset columns and their tiers |
| `data/TRMM_trackwise_decisions.parquet` | enriched dataset, 352,800 decisions x 35 columns (after the run; not in git) |
| `data/enrich_report.json` | row/column counts, sha256, NaN and validity counts (after the run) |
| `results/` | baseline experiments E1-E11 (after the run) |
| `exports/`, `cesium/` | CSV and Cesium CZML exports (after the run) |
| `TRMM_trackwise.ipynb` | analysis notebook, built but not executed until the dataset exists |

Run, from any working directory (the cache folder is created there). The extraction needs the
interpreter that has TATC 3.4.10 (the `trackwise-py` kernel's Python,
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`); Homebrew's `python3` does not:

```
cd ~/trackwise_run && nohup caffeinate -is /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    -u "<repo>/extract_trackwise.py" --mission trmm > trmm.log 2>&1 &                             # ~30-40 h
python3 task_outputs/enrich_trackwise.py --mission trmm ~/trackwise_run/cache_trmm_8mo_jun2005
python  -m baselines.run_baselines --mission trmm --mlp
python3 task_outputs/export_trackwise.py --mission trmm                                           # optional
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=trackwise-py \
    missions/trmm/TRMM_trackwise.ipynb
```

## Seasonal robustness evaluation

The historical CSVs in `results/` are preserved as the **Chronological future-season stress test**
(June–September train, October validation, November–January test including the existing February
tail). The primary seasonal evaluation lives separately in `results/seasonal_lomo/`.

```sh
python -m baselines.run_baselines --protocol seasonal --mission trmm
python -m baselines.paper_figures --protocol seasonal --mission trmm
```

The seasonal command includes P1–P8, including MLP and oracle reference, and records eight G5NR
weather-month folds with February merged into January. Preprocessing and thresholds use only the
other seven months. Every score includes nominal and realized duty. See
`docs/baseline_paper/07_experiments.md` for tests, smoke runs, provenance, resuming and output definitions.
The notebook retains the historical section and displays the new monthly results separately.
