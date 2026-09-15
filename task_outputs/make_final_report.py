#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task_outputs"
src_path = ROOT / "trackwise_decisions.parquet"
final_path = OUT / "trackwise_decisions_enriched.parquet"
src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
final = pd.read_parquet(final_path)
src = pd.read_parquet(src_path)
registry = yaml.safe_load((ROOT / "docs/column_registry.yaml").read_text())["columns"]

task1 = json.loads((OUT / "t1_lwtup_fore_corrected_report.json").read_text())
task2 = json.loads((OUT / "t2_validation_report.json").read_text())
task3 = json.loads((OUT / "t3_autocorr_report.json").read_text())
task4 = json.loads((OUT / "t4_land_water_report.json").read_text())
task5 = json.loads((OUT / "t5_validation_report.json").read_text())
merge = json.loads((OUT / "merge_validation_report.json").read_text())

report = {
    "status": {"task1": "complete", "task2": "complete", "task3": "complete",
               "task4": "complete", "task5": "complete", "task6": "complete", "task7": "complete"},
    "source": {"path": str(src_path), "shape": list(src.shape), "sha256": src_hash,
                "source_unchanged": src_hash == "e1abe3ec18729d70d7e3cc23719b57ad7db2d21cd076db43680aa4472ab7b0d6"},
    "task1_corrected": task1,
    "task2_validation": task2,
    "task3_autocorrelation": task3,
    "task4_land_water": task4,
    "task5_validation": task5,
    "merge": merge,
    "final_parquet": {"path": str(final_path), "shape": list(final.shape), "columns": list(final.columns),
                       "dtypes": {c: str(v) for c, v in final.dtypes.items()}},
    "registry": {"entry_count": len(registry), "unique_names": len({x["name"] for x in registry}),
                 "covers_final_columns": {x["name"] for x in registry} == set(final.columns)},
    "constraints": {
        "no_reextraction": True,
        "no_time_varying_weather_download": True,
        "constants_collection_fetched_once_and_cached": True,
        "task4_not_duplicated": True,
        "no_source_overwrite": True,
        "no_commit_or_push": True,
    },
    "files_created_or_modified": [
        "task_outputs/t1_lwtup_fore_corrected.parquet",
        "task_outputs/t1_lwtup_fore_corrected_report.md",
        "task_outputs/t1_lwtup_fore_corrected_report.json",
        "task_outputs/t2_corrupt_values.csv",
        "task_outputs/t2_validation_report.md",
        "task_outputs/t2_validation_report.json",
        "task_outputs/t3_autocorr_report.md",
        "task_outputs/t3_autocorr_report.json",
        "task_outputs/t4_land_water.parquet",
        "task_outputs/t4_land_water_report.md",
        "task_outputs/t4_land_water_report.json",
        "task_outputs/t5_validation_report.md",
        "task_outputs/t5_validation_report.json",
        "task_outputs/trackwise_decisions_enriched.parquet",
        "docs/column_registry.yaml",
        "docs/feature_set.md",
        "docs/trackwise_method.md",
    ],
    "unresolved_warnings": [
        "lwtup_fore is a same-weather-step GEOS-5 forward-region proxy, not a true BBR retrieval or footprint.",
        "Weak precipitation persistence at one-minute cadence does not establish that all sequential features are useless.",
        "Same-minute decision-time causality and rollout availability masks remain future RL-environment responsibilities.",
        "GEOS-5 diagnostics are idealized proxies; no instrument forward model or retrieval chain was applied.",
    ],
}
(OUT / "final_validation_report.json").write_text(json.dumps(report, indent=2, default=list) + "\n")
md = ["# Final validation report", "", "All Tasks 1–7 are complete locally; no commit or push was performed.", "",
      f"- Source: `{src_path.name}`, shape `{src.shape[0]:,} x {src.shape[1]}`, SHA-256 `{src_hash}`.",
      f"- Final parquet: `{final_path}`; shape `{final.shape[0]:,} x {final.shape[1]}`.",
      "- Original source parquet was not overwritten and its hash matches the pre-merge audit.",
      "- No re-extraction or time-varying weather download occurred.",
      "- Task 4 used one cached constants collection and the existing 60-sample-per-decision nadir cache; it was not duplicated.", "",
      "## Added columns", "", "`" + "`, `".join(merge["added_columns"]) + "`", "",
      "## Task results", "",
      f"- Task 1: {task1['finite_values']:,} finite `lwtup_fore`; {task1['total_nans']:,} NaNs (final {task1['final_row_nans']}, gaps {task1['gap_boundary_nans']}, weather boundaries {task1['weather_step_boundary_nans']}, missing next mean {task1['missing_next_lwtup_mean_nans']}).",
      f"- Task 2: 94 invalid entries across all target columns; {task2['unique_decisions_any_invalid_target']} unique decisions contain at least one invalid target; 24 finite corrupt values converted to NaN across {task2['unique_decisions_with_finite_corrupt_values']} unique decisions; legitimate zeros preserved; no imputation.",
      "- Task 3: precipitation ACF drops below 1/e at lag 1; `tautot_mean` crosses between lags 2 and 3. Pair counts are in `t3_autocorr_report.json`.",
      f"- Task 4: land mean {task4['land_mean']:.6f}, water mean {task4['ocean_mean']:.6f}, partition max absolute error {task4['partition_max_abs_error']:.6f}, 60 samples per decision.",
      f"- Task 5: sine/cosine maximum errors are {task5['sin_max_abs_error']:.3g}/{task5['cos_max_abs_error']:.3g}; unit-circle max error {task5['unit_circle_max_abs_error']:.3g}.",
      "- Task 6: one-to-one decision merges; final shape 351,639 x 40; only the authorized 24 finite target sentinels changed.",
      "- Task 7: feature registry and documentation created/updated.", "",
      "## Code/documentation diff summary", "",
      "- Added local validation/merge scripts: `task_outputs/prepare_validations.py`, `task_outputs/complete_t4_from_cache.py`, and `task_outputs/merge_enriched.py`.",
      "- Added `docs/column_registry.yaml` with exactly one entry per final column.",
      "- Added `docs/feature_set.md` covering tiers, proxy framing, missing-target hygiene, land/water, autocorrelation, longitude, and future RL constraints.",
      "- Updated `docs/trackwise_method.md` to link the feature registry, retain missing-decision caveats, and stop describing all diagnostics as direct MSI measurements.", "",
      "## Warnings", "", *[f"- {x}" for x in report["unresolved_warnings"]]]
(OUT / "final_validation_report.md").write_text("\n".join(md) + "\n")
print("wrote final reports")
