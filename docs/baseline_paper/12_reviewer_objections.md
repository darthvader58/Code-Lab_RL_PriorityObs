# 12. Anticipated Objections and Responses

## “Is poor winter performance a weak model or seasonal distribution shift?”

The original chronological protocol cannot distinguish those causes by itself. We retain it as
the **Chronological future-season stress test**, and use eight seasonal LOMO folds as the primary
robustness evaluation for EarthCARE and TRMM separately. Each fold trains on the other seven G5NR
weather months, merges the partial February tail into January, and purges adjacent weather steps.
Preprocessing and thresholds are fitted on training-side data only.

Every held-out month is reported, including base rate, nominal and realized duty, AUC/AP, capture,
mass capture and precision; summaries give mean, SD, minimum and maximum. Label geometry, feature
tiers, geographic strata, rare labels, missingness, bootstrap uncertainty and feature importance
are repeated per fold. E12 matches months and latitude range. The protocols answer different
questions and have different training sizes/calibration: their difference is not a pure seasonal
effect. Eight simulated months do not establish general all-season performance. See generated
[seasonal results](16_seasonal_results.md) and the [protocol](07_experiments.md).

The answers below retain historical chronological examples; any unqualified historical performance
number must be read within that stress-test scope.


Use these to pre-empt reviewers in the text (each response says where it belongs) and to prepare
for advisor and conference questions.

### O1. "This is just dynamic targeting, which JPL has already done and flown."

**Response:** Agreed that onboard targeting, learning-based targeting policies, and flight
demonstration all exist [breitfeld2025learning, dt2025cognisat6]. This work studies a different
variant: on/off gating of an active nadir radar under a duty-cycle budget, using a co-located imager
rather than a lookahead sensor, scored against the radar's own footprint. **Where:** §2.6 table and
§2.7 gap statement.

### O2. "EarthCARE's CPR runs continuously. Why would anyone gate it?"

**Response:** EarthCARE is the geometric testbed, not the target. The budget models future
power- or data-limited radars, e.g., small-satellite Ka-band radars of RainCube heritage such as
those planned for INCUS [peral2019raincube, tanelli2023incus]. **Where:** §1.1; Limitation 4.
Before submission, quote a verified power or duty-cycle figure for one such mission.

### O3. "Satellites don't have the compute to run an RL agent." (advisor's objection)

**Response:** Two costs must be separated. The trained policy costs ~15 to ~5,000 operations per
decision once a minute (16 to 18,300 parameters, 3 kB to 1 MB on EarthCARE) — negligible even for
radiation-hardened processors, and onboard neural inference has already flown [giuffrida2022phisat].
On EarthCARE, cheap thermal-infrared-like inputs keep 90% of the full-feature capture. Timing was
measured on a laptop, not on flight-like hardware; the paper states this as an assumption
(Limitation 11) and lists hardware timing as future work. The real constraint is producing the input
features; Tier B/C measure how much skill survives without onboard retrievals. Training happens on
the ground. **Where:** §6, Table 5.

### O4. "The features are model diagnostics, not MSI retrievals."

**Response:** Correct and stated explicitly. They are idealized proxies; results are an upper bound
on what a real imager provides. Tier B and the night analysis bound the effect, and validation on
real EarthCARE MSI/CPR data is the first follow-up. **Where:** §3.5, Limitation 1, §8.2.

### O5. "Your AUC is inflated by temporal leakage."

**Response:** Primary seasonal results use weather-month holdouts with buffers on both sides of
each boundary. Fitting and calibration exclude the held-out month. E1 retains the random versus
chronological comparison, but the comparison changes both local temporal dependence and seasonal
composition; a small score difference does not establish absence of leakage. **Where:** §7 E1/E2,
§16 generated monthly results.

### O6. "One year of one model; does it generalise?"

**Response:** Leave-one-month-out results across eight months spanning boreal summer to winter;
known G5NR biases (split ITCZ, weaker Hadley circulation [gelaro2015g5nr]) are discussed. Real-data
validation is future work. Three missions with different orbits, radar geometries, and passive
sensors show whether conclusions hold beyond one geometry, though all three use the same simulated
period. **Where:** E2, E12, §10.5, §10.8, Limitations 3 and 7.

### O7. "Why not just train RL now?"

**Response:** Without a fixed metric, split, and static baseline at the same budget, an RL result
cannot be interpreted. With lag-1 precipitation autocorrelation of only 0.11 on EarthCARE, a
sequential policy must earn its gain from budget dynamics or revealed features, which this benchmark
makes testable. The bar is now fixed: 66.4% capture at 20% duty (AUC 0.893).
**Where:** §1.3, §10.7.

### O8. "The 7 km model can't represent a ~1 km radar footprint."

**Response:** Correct; the nadir label is the maximum over model cells under 60 one-second
sub-satellite points, not a radar simulation. It is still far closer to the instrument footprint
than swath truth, which is the comparison the label finding makes. **Where:** §4.2, Limitation 2.

### O9. "Why 1 mm/hr? Why not the median split?"

**Response:** 1 mm/hr is a physically meaningful threshold that keeps the realistic minority class
the budget problem depends on; the median split and other thresholds are reported as sensitivity.
**Where:** §5.2, E7.

### O10. "Is the gain just geography — firing in the tropics?"

**Response:** That is exactly what the geometry-only and latitude-band baselines (P2, P3) measure;
the imager's value is reported as the margin over them: on EarthCARE, 27.5–28.0% capture at 20% duty
for geometry vs. 66.4% with the passive features. **Where:** §5.4, Table 4.

### O11. "A sun-synchronous orbit only sees two times of day. Doesn't that bias the results?"

**Response:** It limits what the results describe, and the paper states this. EarthCARE samples
~14:00 and ~02:00 local time, so every reported rate is conditional on precipitation at those times,
not on daily precipitation. The labels remain correct for the task, because a radar can only gate
what is beneath it when it passes over. The open question is transfer to orbits with different local
times. The paper measures this directly: TRMM and FY-3G sample all local times, and E12 compares
TRMM's skill near 14:00/02:00 with its skill at other hours and tests transfer from EarthCARE. G5NR
also has every half hour, so the fraction of daily precipitation EarthCARE's two windows miss can be
measured, which real EarthCARE data cannot do. **Where:** §3.3; Limitation 8; E12; §8.2 future
work 5.

### O12. "Why three missions? Why not one model trained on all of them?"

**Response:** The missions differ in orbit (sun-synchronous vs. precessing; ±82°, ±35° and ±50°
reach), radar geometry (nadir line vs. 247 or 303 km scanning swath), and base rate. One pooled model would
have to drop to the features they share and would hide exactly these differences. Each mission is
therefore a separate case with its own dataset, training, and validation; transfer between missions
is measured separately (E12). If a reviewer prefers depth over breadth: EarthCARE is the primary case
and is reported in full, and the other two use the identical method. **Where:** §1.2, §5.9, E12,
§8.3, Limitation 7.

### O13. "TRMM and FY-3G carry microwave imagers. Why not use them?"

**Response:** The nature run does not produce microwave brightness temperatures, so using TMI or
MWRI-RM would need real data or a microwave forward model. A microwave imager also already sees rain,
which turns the gate into a different question (what the radar adds beyond a sensor that sees rain)
needing a different label. Using the visible/infrared imager on every mission keeps the passive
context the same in kind, so the comparison isolates orbit and radar geometry. The microwave case is
stated as future work. **Where:** §3.8, §5.8, §8.2 future work 8.

### O14. "TRMM only covers the tropics. Is the comparison fair?"

**Response:** Cross-mission comparisons are restricted to shared latitudes or reported by latitude
band, and TRMM's results are described as tropical and subtropical only. TRMM is included because it
flew during G5NR's own period and its visible/infrared scanner matches EarthCARE's imager closely,
which isolates the effect of the orbit. **Where:** §3.7, E12, Limitation 9.

### O15. "You train on summer and test on winter. Isn't the test unfair?"

**Response:** The chronological test intentionally measures future-season transfer, not neutral
all-season skill. LOMO is the primary seasonal robustness analysis: each omitted month is scored
after training on the other seven. Report every month and budget drift for both missions. A
protocol difference also changes training size, calibration and test composition, so it is not a
pure seasonal effect or a justified upper bound. Similar rankings do not make one protocol a
conservative substitute for the other. **Where:** §7, §10, §16.
