# Final validation report

All Tasks 1–7 are complete locally; no commit or push was performed.

- Canonical parquet: `trackwise_decisions.parquet`, shape `351,639 x 40`, SHA-256 `4b92f01e24e231a5b5a1151060fef586f08e450e15c252fb23ef9ef772fd1466`.
- The validated enriched data replaced the former 27-column source intentionally; the staging copy remains byte-identical.
- No re-extraction or time-varying weather download occurred.
- Task 4 used one cached constants collection and the existing 60-sample-per-decision nadir cache; it was not duplicated.

## Added columns

`lwtup_fore`, `land_frac_nadir`, `ocean_frac_nadir`, `sin_lon`, `cos_lon`, `prectot_swath_valid`, `prectot_nadir_valid`, `preccon_swath_valid`, `preccon_nadir_valid`, `iwp_swath_valid`, `iwp_nadir_valid`, `lwp_swath_valid`, `lwp_nadir_valid`

## Task results

- Task 1: 339,859 finite `lwtup_fore`; 11,780 NaNs (final 1, gaps 49, weather boundaries 11717, missing next mean 13).
- Task 2: 94 invalid entries across all target columns; 51 unique decisions contain at least one invalid target; 24 finite corrupt values converted to NaN across 15 unique decisions; legitimate zeros preserved; no imputation.
- Task 3: precipitation ACF drops below 1/e at lag 1; `tautot_mean` crosses between lags 2 and 3. Pair counts are in `t3_autocorr_report.json`.
- Task 4: land mean 0.338833, water mean 0.660837, partition max absolute error 0.000754, 60 samples per decision. The 17.09% mixed-surface statistic is defined as `0.05 < land_frac_nadir < 0.95`; it is not necessarily individual shoreline cells. The orbit-sampled land mean is plausible for a near-polar track but is not directly comparable to an area-weighted global land fraction because orbital sampling is not uniform by surface area.
- Task 5: sine/cosine maximum errors are 0/0; unit-circle max error 2.22e-16.
- Task 6: one-to-one decision merges; final shape 351,639 x 40; only the authorized 24 finite target sentinels changed.
- Task 7: feature registry and documentation created/updated.

## Code/documentation diff summary

- Added local validation/merge scripts: `task_outputs/prepare_validations.py`, `task_outputs/complete_t4_from_cache.py`, and `task_outputs/merge_enriched.py`.
- Added `docs/column_registry.yaml` with exactly one entry per final column.
- Added `docs/feature_set.md` covering tiers, proxy framing, missing-target hygiene, land/water, autocorrelation, longitude, and future RL constraints.
- Updated `docs/trackwise_method.md` to link the feature registry, retain missing-decision caveats, and stop describing all diagnostics as direct MSI measurements.

## Warnings

- lwtup_fore is a same-weather-step GEOS-5 forward-region proxy, not a true BBR retrieval or footprint.
- Weak precipitation persistence at one-minute cadence does not establish that all sequential features are useless.
- Same-minute decision-time causality and rollout availability masks remain future RL-environment responsibilities.
- GEOS-5 diagnostics are idealized proxies; no instrument forward model or retrieval chain was applied.
