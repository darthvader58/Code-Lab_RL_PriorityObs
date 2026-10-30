# 7. Experiments: seasonal robustness and future-season transfer

EarthCARE and TRMM are separate benchmarks with unchanged datasets, features, mission labels,
footprints and model architectures. They use the same evaluation implementation.

**Primary evaluation: seasonal leave-one-month-out (LOMO).** This answers: does the model remain
reliable across the sampled seasonal weather regimes?

**Secondary evaluation: Chronological future-season stress test.** This answers: how well does a
model prepared on earlier months transfer into a later unseen season? Preserve the existing
June–September training, October validation and November–January test, including the partial
February tail already present in the historical test. This is not a neutral all-season benchmark.

The two protocols estimate different quantities. They also differ in training size and threshold
calibration. Their score difference is not an isolated causal estimate, or a guaranteed upper bound,
of a seasonal penalty. Random-versus-chronological E1 likewise changes both temporal proximity and
seasonal composition; it does not identify leakage alone.

## Shared seasonal fold protocol

`baselines.core.seasonal_folds` defines eight groups using **G5NR weather month**, never orbital
calendar time. The partial February 2006 period joins January 2006. For each held-out month:

1. Train on the other seven weather months, before and/or after the held-out period.
2. Drop the first and last weather steps of the held-out block and the immediately adjacent
   training steps (`lo-1`, `lo`, `hi`, `hi+1`). Define these boundaries before label filtering or
   geographic restriction. Apply the same exclusion to every experiment.
3. Fit imputation medians, missing-indicator selection, scaling and models only on training rows.
   Keep the original architectures and fixed hyperparameters. The MLP's existing internal early
   stopping is confined to training months; it is not an independent month-held-out calibration.
4. Calibrate gates on **training-month score quantiles**, matching the original E2, without
   held-out labels or scores. This is in-sample calibration, not an inner validation split. Its
   possible optimism and poor budget transfer are measured through realized held-out duty.
5. Evaluate each month separately. Report equal-weight monthly mean, sample standard deviation,
   minimum and maximum. Do not substitute a pooled score for the per-month results.

P8 uses held-out truth only as an explicitly unattainable oracle reference. It is never fitted or
used to tune a deployable policy. All other policies use training-side threshold selection.

## Experiment definitions and output files

Outputs are under `missions/<mission>/results/seasonal_lomo/`; historical CSVs in the parent
`results/` directory remain unchanged. Each fold also has its own directory and completion hash
manifest. `run_meta.json` records dataset SHA-256, seed, code hashes, versions, fold definitions,
threshold source and run settings. Standard score tables record mission, policy, held-out month,
training months, sample counts, base rate, nominal budget, realized duty, threshold, AUC, average
precision, event capture, mass capture and precision.

| ID | seasonal analysis | output |
|---|---|---|
| E1 | Keep the historical random-versus-time-blocked comparison. Interpret it as affected by both short-range dependence and seasonal shift. | parent `e1_split.csv` |
| E2 | P1 random, P2 geometry LR, P3 latitude band, P4 single threshold, P5 LR, P6 LightGBM, P7 MLP, P8 oracle; every month and fixed budgets 5, 10, 13, 20, 30, 50%, plus training-base-rate budget. | `e2_monthly.csv`, `e2_summary.csv` |
| E3 | Independent capture-versus-nominal-budget curves for every month and policy. Read together with realized duty. | `e3_capture_curves.csv` |
| E4 | Repeat total and convective precipitation train-label × score-label grids on common valid rows in each fold, using the original mission geometries. Geometry-specific mass metrics weight by the scored precipitation column, as in the historical E4. | `e4_label_geometry.csv` and summary |
| E5 | Repeat feature tiers, day/night and switched policies within each fold. Optical-depth tiers are explicitly unavailable at night; do not invent night inputs. Day/night thresholds use their respective training subsets. Switched policies use one training-fold threshold. | `e5_tiers.csv` and summary |
| E6 | Latitude and land/ocean/mixed strata, using P6's **global training-fold threshold**. No stratum-specific retuning. Mass capture uses each mission's primary mass column. | `e6_strata.csv` and summary |
| E7 | Refit LR and LightGBM for 0.1, 1, 5 mm/hr, convective 1 mm/hr and strict training-median labels; report fixed budgets and the training-label-base-rate budget. Thresholds use training-side data only. Sensitivity mass capture weights the corresponding target column, matching the original E7 computation; primary E2/E3 mass capture retains the mission mass column. | `e7_label_sensitivity.csv`; summary at nominal 20% |
| E8 | Per-fold training, held-out and buffered row counts; feature missingness/availability, invalid targets and training missing-indicator eligibility. | `e8_missing_data.csv` |
| E9 | Retain original parameter, operation, serialized-size and inference-time measurements. Seasonal refitting does not warrant a duplicate architecture-cost analysis. | parent `e9_policy_cost.csv` |
| E10 | At least 1,000 whole-G5NR-day bootstrap draws **within each held-out month**, using fixed gates. Shared draws provide paired P6-minus-other comparisons, with P6 prespecified. Report AUC, AP, capture, mass capture, precision and duty. | `e10_bootstrap.csv`, paired and summary files |
| E11 | P6 prespecified for permutation importance; repeat in every held-out month. Single-feature ranking direction is chosen using training AUC, then frozen for held-out AUC/AP. Report monthly importance and its mean/SD/min/max. | `e11_permutation_importance.csv`, `e11_single_feature_auc.csv` and summaries |
| E12 | Match held-out weather months and restrict fitting/calibration/testing to the common latitude domain, `|lat| <= 35.2°`. Fit mission-specific policies separately. Cross-mission transfer uses shared features and target-side other-seven-month **unlabeled** calibration scores. Report own-label and common-nadir comparisons separately. Local-time strata keep the global fold threshold. | `missions/comparison/results/seasonal_lomo/` |

Undefined metrics (e.g. AUC with one class, precision with no firings) remain missing rather than
being replaced with an artificial success or failure. Non-fittable training variants record a
status. P3 uses the training-wide base rate for any unseen latitude band.

E10 summaries of interval endpoints across folds are descriptive. They are **not** confidence
intervals for an all-season pooled mean: folds share training data, and the weather is one model
realization. The P8 bootstrap resamples the reference gate fixed on the original held-out month;
it does not re-optimize the oracle on each draw.

## Running and reproducing

Use an environment matching the versions recorded in the historical `run_meta.json` (the local
baseline environment is `/opt/anaconda3/bin/python`). From the repository root:

```sh
python -m unittest discover -s tests -v
python -m baselines.run_baselines --protocol seasonal --mission earthcare --smoke --n-boot 4 --out /tmp/earthcare-seasonal-smoke
python -m baselines.run_baselines --protocol seasonal --mission trmm --smoke --n-boot 4 --out /tmp/trmm-seasonal-smoke
python -m baselines.run_baselines --protocol seasonal --mission earthcare
python -m baselines.run_baselines --protocol seasonal --mission trmm
python -m baselines.compare_missions --protocol seasonal
python -m baselines.paper_figures --protocol seasonal --mission earthcare
python -m baselines.paper_figures --protocol seasonal --mission trmm --report
python -m baselines.validate_seasonal
python task_outputs/execute_seasonal_notebooks.py
```

Seasonal runs include P7 automatically. `--resume` accepts only matching provenance and verified
completed-fold file hashes. Use a new `--out` directory for a changed protocol or dataset. Smoke
outputs are explicitly marked and cannot be rendered as paper figures. Running this protocol
requires no new extraction, enrichment, or changes to the datasets.

The old `e2_lomo` function remains a compatibility view for reproducing the original P5–P7 table;
it uses the central fold implementation. The new E2/E3 runner evaluates all P1–P8 from the same
folds and predictions. Current numerical evidence and captions are generated in
[16_seasonal_results.md](16_seasonal_results.md), not typed into the manuscript by hand.

The notebook update executes only the tagged seasonal cells and preserves historical outputs;
it does not run extraction or CZML exports. If its notebook kernel differs from the baseline
environment, set `BASELINE_PYTHON` to the baseline interpreter. The local existing
`/opt/anaconda3/bin/python` is the fallback when the kernel lacks LightGBM.
