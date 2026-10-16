# Baseline Paper — Master Outline

**Working title:** *Can a Passive Imager Tell a Nadir Radar When to Fire? A Track-Wise Benchmark for
Budget-Constrained Radar Gating*

**Paper type:** dataset + baselines. No reinforcement learning is trained in this paper; it defines the
benchmark, the metric, and the bar a future RL policy must beat.

This file is the single-page summary of everything the paper will discuss. Each section is
expanded in its own file so it can be pasted into Overleaf section by section:

| file | paper section |
|---|---|
| [`00_abstract.md`](./00_abstract.md) | Title options, abstract, keywords, plain-language summary |
| [`01_introduction.md`](./01_introduction.md) | Introduction, motivation, contributions |
| [`02_related_work.md`](./02_related_work.md) | Related work (condensed review) |
| [`03_dataset.md`](./03_dataset.md) | The track-wise dataset |
| [`04_label_finding.md`](./04_label_finding.md) | Nadir vs. swath labels |
| [`05_baselines.md`](./05_baselines.md) | Gating policies and the evaluation metric |
| [`06_onboard_feasibility.md`](./06_onboard_feasibility.md) | Feature tiers and onboard cost |
| [`07_experiments.md`](./07_experiments.md) | Experiments to run before submission + result tables |
| [`08_limitations_future_work.md`](./08_limitations_future_work.md) | Limitations and the path to RL |
| [`09_references.md`](./09_references.md) | Reference list with verification status |
| [`10_results_and_discussion.md`](./10_results_and_discussion.md) | Results/discussion template with sentence frames per outcome |
| [`11_figures_and_tables.md`](./11_figures_and_tables.md) | Every figure and table, its data source, and status |
| [`12_reviewer_objections.md`](./12_reviewer_objections.md) | Anticipated objections (incl. the onboard-compute one) and responses |
| [`13_key_numbers.md`](./13_key_numbers.md) | Every number in the paper with its source file |
| [`14_submission_plan.md`](./14_submission_plan.md) | Venues, steps to submission, Overleaf section map |

**Reading order for writing:** this file → `13_key_numbers.md` → sections 00–08 → run
`07_experiments.md` → `10_results_and_discussion.md` → `12_reviewer_objections.md` →
`14_submission_plan.md`.

Numbers marked **[computed]** come from the committed dataset
(`trackwise_decisions.parquet`, SHA-256 `4b92f01e…1466`). Numbers marked **[TO RUN]** do not exist
yet and must come from the experiments in `07_experiments.md`. Do not submit with any **[TO RUN]**
left.

---

## 1. Problem statement

An active, nadir-pointing radar (EarthCARE's Cloud Profiling Radar, CPR, ~308 W) is far more
expensive to run than the passive imager beside it (MSI). If a future radar mission could only run
its radar for a fraction *d* of each orbit, which minutes should it choose? The decision has to be
made onboard, once per minute, using only information the imager provides at that moment.

## 2. Research questions

- **RQ1 — Predictability.** How well does imager-visible context predict precipitation in the
  radar's *nadir* footprint, when the evaluation is free of temporal leakage?
- **RQ2 — Gating value.** At a duty-cycle budget *d*, what fraction of nadir precipitation does an
  imager-driven gate capture, compared with random gating, geometry-only gating, and the oracle?
- **RQ3 — Label geometry.** How much does labelling against the imager swath, instead of the radar
  nadir, distort the positive rate and the apparent skill of a gating policy?
- **RQ4 — Onboard feasibility.** How much of the skill survives when the policy is limited to
  features that are cheap to compute onboard, and how much survives at night?

## 3. Contributions

1. **A track-wise benchmark.** 351,639 one-minute decisions along EarthCARE's real ground track
   (TLE-propagated, 1 s sampling), June 2005 – January 2006 of the GEOS-5 Nature Run (G5NR) at
   0.0625° / 30 min. Every row pairs imager-visible context with nadir and swath truth. Feature tiers,
   validity flags, and a hash are published. **[computed]**
2. **A label finding.** Swath labels more than double the positive rate (28.9% vs. 13.0% at
   ≥1 mm/hr), and 55.2% of swath-positive minutes have no qualifying precipitation under the radar.
   **[computed]** The effect on apparent policy skill is **[TO RUN]**.
3. **Budget-constrained baselines.** Capture-vs-duty-cycle curves for random, geometry-only,
   single-threshold, logistic-regression, gradient-boosting, and oracle gates, under a time-blocked
   split. **[TO RUN]**
4. **Onboard feasibility.** The same curves for three feature tiers, reported for day and night,
   with the arithmetic and memory cost of each policy. **[TO RUN]**

## 4. The dataset (→ `03_dataset.md`)

- **Unit:** one row per one-minute CPR on/off decision; 351,639 rows; 1,161 missing decision IDs
  across 49 discontinuities (kept, never bridged).
- **Geometry:** real EarthCARE orbital elements, sub-satellite point every second, ~394 km altitude,
  150 km MSI swath.
- **Weather:** G5NR at 0.0625° (~7 km) and 30 min; 11,731 distinct weather states; window chosen as
  the wettest two-month start in the 24-month record.
- **Extraction:** per-decision bounding-box fetch, reduced to summary values and discarded (avoids a
  ~10 TB full download).
- **Columns (40):** 12 always-available features, 5 radar-conditional features, nadir and swath
  targets (`prectot`, `preccon`, `iwp`, `lwp`), metadata, and 8 validity flags.
- **Quality control:** 710 corrupted fill-value readings removed (0.20%); 24 finite corrupt target
  values set to NaN; no imputation.

## 5. The label finding (→ `04_label_finding.md`)

| label at ≥ 1 mm/hr | positive rate |
|---|---|
| `prectot_nadir` (what the radar can measure) | **12.97%** |
| `prectot_swath` (anywhere in the imager swath) | **28.93%** |
| `preccon_nadir` (convective only) | 1.03% |
| `preccon_swath` | 4.45% |

Swath-positive minutes with no nadir precipitation ≥ 1 mm/hr: **55.2%** of all swath positives.
The reverse never happens (swath is the maximum over a superset of the nadir cells).
Takeaway: any targeting study that scores a nadir instrument against imager-scale truth more than
doubles its positive class and rewards the policy for minutes the instrument could not have measured.

## 6. The baselines (→ `05_baselines.md`)

- **Primary label:** `prectot_nadir ≥ 1 mm/hr`. The existing median-split `fire` label (~0.09 mm/hr)
  is a sensitivity check, not the headline.
- **Primary metric:** *capture@d*, the fraction of positive minutes the gate fires on when it may
  fire on a fraction *d* of minutes. Also: precipitation-weighted capture, precision-at-budget,
  ROC-AUC, average precision.
- **Policies:** random; always-on (upper bound with no budget); geometry-only; single-feature
  threshold on `tautot_mean`; logistic regression; gradient-boosted trees; oracle.
- **Threshold rule:** each learned score is turned into a gate by picking the threshold that fires
  on exactly a fraction *d* of validation minutes.

## 7. Onboard feasibility (→ `06_onboard_feasibility.md`)

- **Tier A (full):** all 12 always-available features. Upper bound; assumes onboard cloud retrievals.
- **Tier B (cheap):** thermal-infrared-like quantities (`cldtmp_mean`, `lwtup_mean`, `lwtup_fore`)
  plus geometry. Works day and night. No optical-depth retrieval.
- **Tier C (geometry):** `lat`, `solar_hour`, `sin_lon`, `cos_lon`, `land_frac_nadir`,
  `ocean_frac_nadir`. Needs no imager at all.
- **Cost:** logistic regression ≈ 12 multiply-adds per decision; a 12-64-64-1 MLP ≈ 5,000; both
  fit in kilobytes. One decision per minute is far inside a flight processor's budget. The real cost
  is producing the features, which is why Tier B exists.

## 8. Experiments to run before submission (→ `07_experiments.md`)

1. Replace the random 30% split with a **time-blocked split** with a buffer between blocks; report
   how much the current 0.837 AUC changes.
2. **Leave-one-month-out** cross-validation for seasonal robustness.
3. **Capture@d** at d ∈ {5, 10, 13, 20, 30, 50}% for every policy.
4. **Nadir vs. swath** training and scoring (2 × 2 grid) to quantify RQ3.
5. **Feature tiers A/B/C**, each reported for day, night, and overall.
6. **Stratified skill:** tropics / mid-latitudes / polar; land / ocean / mixed.
7. **Label-threshold sensitivity:** 0.1, 1, 5 mm/hr and the median split; convective-only target.
8. **Missing-data handling:** rows with invalid features or targets excluded and counted.
9. **Onboard cost table:** operations, parameters, and memory per policy.
10. **Statistical uncertainty:** block bootstrap confidence intervals on capture@d.

## 9. Limitations (→ `08_limitations_future_work.md`)

G5NR diagnostics are idealized proxies, not MSI retrievals; ~7 km vs. ~500 m resolution; one
simulated period (2005–06) from a model with a known split-ITCZ bias; EarthCARE's real CPR runs continuously, so the budget is a stand-in for a
future power-limited radar; decisions are scored independently (no battery or reveal-on-fire
dynamics yet). Follow-ups: validation on real EarthCARE MSI + CPR data, and sequential RL that must
beat these baselines.

## 10. One-sentence claim of the paper

> With only imager-visible context, a one-minute gate can capture **[TO RUN]%** of nadir
> precipitation while the radar runs **[TO RUN]%** of the time; a cheap, retrieval-free feature
> set keeps **[TO RUN]%** of that skill; and scoring against swath rather than nadir truth would
> have more than doubled the positive class.
