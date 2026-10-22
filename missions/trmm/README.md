# TRMM (NORAD 25063)

35° precessing orbit at ~400 km (all local times), scanning Precipitation Radar (PR, 13.8 GHz,
247 km swath), Visible and Infrared Scanner (VIRS, 833 km swath). Detection-prioritization
benchmark: should the PR be on this minute, judged from VIRS-like context?

**Status:** code complete and checked offline; the extraction has not been run yet.
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

Run, from any working directory (the cache folder is created there):

```
cd ~/trackwise_run && nohup python3 -u "<repo>/extract_trackwise.py" --mission trmm > trmm.log &   # ~30-40 h
python3 task_outputs/enrich_trackwise.py --mission trmm ~/trackwise_run/cache_trmm_8mo_jun2005
python  -m baselines.run_baselines --mission trmm --mlp
python3 task_outputs/export_trackwise.py --mission trmm                                           # optional
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=trackwise-py \
    missions/trmm/TRMM_trackwise.ipynb
```
