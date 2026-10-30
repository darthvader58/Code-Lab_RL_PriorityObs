# 10. Results and Discussion

The primary seasonal robustness evidence is the generated, per-mission report
[16_seasonal_results.md](16_seasonal_results.md), sourced from
`missions/earthcare/results/seasonal_lomo/` and `missions/trmm/results/seasonal_lomo/`.
The historical numerical tables in [13_key_numbers.md](13_key_numbers.md) and
[15_earthcare_results.md](15_earthcare_results.md) describe the **Chronological future-season stress
test**, except their explicitly labeled legacy E2 month-held-out tables. They are retained as
historical evidence, not substituted for the new seasonal results.

## 10.1 Distinguish the two scientific questions

The chronological test asks how well a model trained on June–September and calibrated on October
transfers to November–January (including the existing partial February tail). It deliberately
mixes model quality with transfer into later-season weather. A fixed November–January result is
not sufficient evidence of all-season performance.

LOMO asks whether the model remains reliable across the eight sampled weather months. A model
is fitted on the other seven months and tested on the omitted month. These folds may train on
future months relative to the held-out month: they estimate seasonal robustness, not operational
forecasting performance. Report every month, plus equal-weight mean, standard deviation and range.
Do not interpret a difference from the chronological score as a causal seasonal penalty: training
size, calibration and evaluation populations also differ.

E1's random split places nearby decisions from shared weather snapshots on both sides of the
split, but its comparison with the chronological test also changes seasonal composition. Similar
scores do not prove absence of leakage, and a decline cannot be attributed exclusively to leakage.

## 10.2 Gating value under seasonal budgets (E2, E3, E10)

Read the monthly capture curves alongside monthly base rate and realized duty. A nominal 20%
training quantile does not enforce a 20% hard operational budget in a new month. Higher event
capture may accompany increased radar use; distinguish ranking quality (AUC/AP), event capture,
mass capture, precision and budget drift. P8 is an unattainable reference with truth available.
Its ceiling depends on the month's base rate, and its primary rain-rate ordering is not necessarily
a mass-optimal ordering for scanning-radar mean-rain weights.

Report within-month whole-day bootstrap intervals, including paired comparisons against the
prespecified P6 reference. An interval endpoint averaged across months is a descriptive summary,
not an all-season confidence interval. Avoid choosing a winning policy from a test month and
then presenting it as a prespecified inference.

## 10.3 Geometry, feature tiers and deployment regimes (E4–E6)

Repeat the footprint/swath train-score comparison for each month: changing storm structure can
change positive-rate inflation and event capture. Keep the mission's actual radar label as the
primary task. In E4, mass metrics follow the scored geometry; in primary E2/E3/E6 results, they
follow the mission's existing mass column.

Compare day, night and switched policies across months. Optical-depth tiers are unavailable for
night-only evaluation, and diagnostic G5NR proxies are not real onboard retrievals. E6 keeps one
global training-fold threshold when reporting each latitude and land/ocean stratum. Duty drift in
a stratum is deployment evidence; retuning locally would hide it. Retain the existing E9 cost
measurements and the limitation that laptop timings are not flight-hardware measurements.

## 10.4 Label difficulty, availability and interpretation (E7, E8, E11)

Compare the 0.1, 1 and 5 mm/hr labels, convective precipitation and training-median labels month by
month. AUC alone can look strong for rare events while precision or capture at a fixed budget
remains poor. Missingness and feature availability are reported separately for training and held-out
rows. Imputation, indicator selection and scaling are fitted exclusively on the training months.

Permutation importance uses P6, selected in advance, in each held-out month. Discuss its monthly
mean and range rather than declaring a single winter feature ranking universal. Single-feature
score directions are fitted on training data, so a reversed association in a new month is retained
as a possible failure instead of being corrected with held-out labels.

## 10.5 EarthCARE versus TRMM (E12)

Use `missions/comparison/results/seasonal_lomo/` for the seasonal comparison: match the omitted
weather month and use `|lat| <= 35.2°` for fitting, calibration and testing in both missions.
EarthCARE and TRMM keep their distinct labels, footprint definitions and fitted policies.
Cross-mission transfer additionally reports a common nadir-label setting and shared features.
The target calibration scores come from the other seven months, never the held-out month.

Report the source and target dataset hashes. The missions sample the same G5NR weather, so this
is not independent meteorological replication. Different footprints, local times and passive
features remain relevant even after matching months and latitudes. Historical E12 chronological
results remain separately labeled; do not mix them into seasonal means.

## 10.6 Scope of the conclusions and the RL follow-up

These experiments measure robustness across a limited set of G5NR model-weather months, not a
complete annual cycle, multiple years, real sensor distributions, or future climates. Base rates,
thresholds, realized duty and feature importance may vary by month and mission.

A future sequential policy should be evaluated under both protocols, with its state/reset rules,
budget constraints and observable inputs explicit. Compare its capture at comparable **realized**
radar use and its monthly worst-case performance as well as its mean. A gap to the oracle is not
a guarantee that RL can close it. The two retained answers remain distinct:

- Future-season transfer: how well does preparation on earlier months transfer into later weather?
- Seasonal robustness: how consistently does a policy work across the sampled weather regimes?
