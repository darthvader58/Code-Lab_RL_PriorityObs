# 7. Experiments — Run All of These Before Submission

Status key: ☐ not started · ◐ partial · ☑ done. Every result table below starts empty; fill it in
from code, not by hand.

**Every experiment E1–E11 is run separately for each mission** (EarthCARE first, then TRMM, then
GPM), on that mission's own dataset, splits, and policies. Each result table below is repeated per
mission. E12 compares the missions. Numbers quoted from the earlier EarthCARE extraction (e.g., AUC
0.837) are **[TO RERUN]** and must not appear in the paper.

## Pre-submission checklist

| ID | experiment | answers | status |
|---|---|---|---|
| E1 | Time-blocked split vs. the existing random 30% split | RQ1, leakage | ☐ |
| E2 | Leave-one-month-out cross-validation | RQ1, seasonal robustness | ☐ |
| E3 | Capture@d curves for all policies P0–P8 | RQ2 | ☐ |
| E4 | Nadir vs. swath 2 × 2 train/score grid | RQ3 | ◐ (positive rates done) |
| E5 | Feature tiers A/B/C × day/night/all | RQ4 | ☐ |
| E6 | Stratified skill by latitude band and surface | RQ1/RQ2 | ☐ |
| E7 | Label-threshold sensitivity (0.1, 1, 5 mm/hr, median split, convective) | robustness | ☐ |
| E8 | Missing-data accounting | reproducibility | ☐ |
| E9 | Policy cost (ops, parameters, memory, wall time) | RQ4 | ☐ |
| E10 | Block-bootstrap confidence intervals | all claims | ☐ |
| E11 | Feature importance / ablation for the best policy | interpretation | ☐ |
| E12 | Cross-mission comparison and transfer | RQ5 | ☐ |

Per-mission status:

| mission | dataset | E1–E11 |
|---|---|---|
| EarthCARE | extraction in progress (season-aligned TLE set) | ☐ |
| TRMM | ☐ TLE set requested? dataset not built | ☐ |
| GPM | ☐ data source (D2) and label (D3) not decided | ☐ |

---

## E1 — Time-blocked split (do this first; everything else depends on it)

**Why:** consecutive decisions in the same 30-minute weather step share a weather field
(`03_dataset.md` §3.6). The earlier 0.837 AUC / 0.368 AP came from a random 30% split and may have been
inflated.

**Protocol:**
- Train: June–September 2005 · Validation: October 2005 · Test: November 2005–January 2006
  (by G5NR weather time, i.e., `g5_step`, not the TLE `time` column).
- Drop at least one full weather step (30 min) at every block boundary as a buffer.
- Report logistic regression on Tier A under both splits.

| split | ROC-AUC | AP | capture@13% |
|---|---|---|---|
| random 30% | [TO RERUN] (earlier extraction: 0.837) | [TO RERUN] (0.368) | [TO RUN] |
| time-blocked | [TO RUN] | [TO RUN] | [TO RUN] |

Note: the time-blocked test set is a different season (NH winter) from training (NH summer). Any
drop mixes leakage removal with seasonal shift — E2 separates them.

## E2 — Leave-one-month-out

Train on 7 months (with buffers), test on the held-out month; repeat for all 8. Report mean ± range.

| held-out month | AUC | AP | capture@13% |
|---|---|---|---|
| Jun 2005 … Jan 2006 | [TO RUN] | | |

## E3 — Capture@d curves

Policies P0–P8 from `05_baselines.md`, Tier A, time-blocked test set.

| policy | AUC | AP | capture@5% | @10% | @13% | @20% | @30% | @50% | mass-capture@20% |
|---|---|---|---|---|---|---|---|---|---|
| P1 random | 0.5 | 0.130 | 5% | 10% | 13% | 20% | 30% | 50% | ~20% |
| P2 geometry LR | | | | | | | | | |
| P3 latitude band | | | | | | | | | |
| P4 `tautot_mean` threshold | | | | | | | | | |
| P5 logistic regression | | | | | | | | | |
| P6 gradient boosting | | | | | | | | | |
| P8 oracle | 1.0 | 1.0 | min(1, 5%/ȳ) | … | | | | | |

(Oracle capture@d = min(1, d / ȳ), with ȳ the mission's base rate on the test block. Random AP = base
rate. Fill the oracle and random rows from the new data for each mission.)

**Figure 1:** capture vs. duty cycle, all policies, with the random diagonal and the oracle curve.

## E4 — Nadir vs. swath grid

| trained on → scored on | nadir | swath |
|---|---|---|
| nadir | [TO RUN] | [TO RUN] |
| swath | [TO RUN] | [TO RUN] |

Report AUC and capture@20%. Repeat for `preccon`. For TRMM and GPM, "nadir" becomes "radar swath"
and "swath" becomes "passive swath" (`04_label_finding.md` §4.4). Earlier EarthCARE positive rates
(nadir ~13% vs. swath ~29%; ~55% of swath positives nadir-negative) are **[TO RERUN]**.

## E5 — Feature tiers × day/night

| tier | day capture@20% | night capture@20% | all (switched) capture@20% | retention vs. A |
|---|---|---|---|---|
| A (full) | | n/a (`tautot` invalid) | | 100% |
| A − `tautot*` | | | | |
| B (cheap) | | | | |
| C (geometry) | | | | |

**Figure 2:** capture@d curves per tier, night only.

## E6 — Stratified skill

Bands: tropics (|lat| < 23.5°), subtropics (23.5–35°), mid-latitudes (35–60°), polar (> 60°).
Surface: ocean (`ocean_frac_nadir` > 0.95), land (`land_frac_nadir` > 0.95), mixed. Report base rate,
AUC, and capture@20% per stratum with the global threshold (do not re-tune per stratum).

## E7 — Label sensitivity

Re-run P5 and P6 (Tier A) on: `prectot_nadir` ≥ 0.1, 1, 5 mm/hr; the median-split `fire` label;
`preccon_nadir` ≥ 1 mm/hr. Report base rate, AUC, AP, capture at *d* = base rate.

## E8 — Missing-data accounting

One table: rows dropped for invalid targets, rows with any NaN feature, how NaN was handled per
policy, final train/val/test sizes.

## E9 — Policy cost

Fill in `06_onboard_feasibility.md` §6.2 with measured operations, parameter counts, serialized
size, and wall time per decision on a constrained CPU.

## E10 — Uncertainty

Block bootstrap over whole days (or weather-step blocks), ≥ 1,000 resamples, 95% CI on AUC, AP,
and capture@d. Plain row bootstrap will understate the uncertainty because rows are correlated.

## E11 — Interpretation

Permutation importance (on the test block) or SHAP for the best policy; single-feature AUCs under
the time-blocked split to update the table in `../trackwise_method.md` §4.3.

---

## E12 — Cross-mission comparison and transfer (RQ5)

Run only after E1–E11 are complete for at least two missions.

1. **Comparison table** at the same budgets: base rate, AUC, capture@d, and Tier B retention per
   mission, side by side. Discuss differences in terms of orbit (local-time sampling), radar footprint
   (nadir vs. scanning), and passive information (cloud-only vs. microwave).
2. **Transfer.** Train on mission A, test on mission B using only the features both share (Tier C and,
   for EarthCARE ↔ TRMM, the cloud proxies). Report capture@d relative to mission B's own policy.

| trained on ↓ / tested on → | EarthCARE | TRMM | GPM |
|---|---|---|---|
| EarthCARE | own policy | [TO RUN] | [TO RUN] (Tier C only) |
| TRMM | [TO RUN] | own policy | [TO RUN] (Tier C only) |
| GPM | [TO RUN] (Tier C only) | [TO RUN] (Tier C only) | own policy |

3. **Local-time test.** Evaluate the TRMM policy separately on decisions near 14:00 and 02:00 local
   time vs. all other hours, and compare with EarthCARE. This measures how much of the day EarthCARE's
   two sampling times leave out (Limitation 8).

Restrict the comparison to latitudes all compared missions cover (TRMM reaches only ~35–40°), or
report it per latitude band.

## Reproducibility requirements

- One script (e.g., `baseline/run_baselines.py --mission earthcare|trmm|gpm`) that regenerates every
  table and figure from that mission's parquet and fixed seeds.
- Record the parquet hash, package versions, and seeds in the output.
- Commit the result tables (CSV/JSON) next to the figures.
