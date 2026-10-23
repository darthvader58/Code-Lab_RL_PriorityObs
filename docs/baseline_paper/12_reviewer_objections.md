# 12. Anticipated Objections and Responses

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

**Response:** All headline results use a time-blocked split with a buffer of at least one weather
step, and the random-split numbers are shown only for comparison. On EarthCARE the split barely
matters (logistic regression AUC 0.833 random vs. 0.837 time-blocked), and leave-one-month-out gives
0.811–0.844. **Where:** §7 E1, E2.

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
times. The paper measures this directly: TRMM and GPM sample all local times, and E12 compares
TRMM's skill near 14:00/02:00 with its skill at other hours and tests transfer from EarthCARE. G5NR
also has every half hour, so the fraction of daily precipitation EarthCARE's two windows miss can be
measured, which real EarthCARE data cannot do. **Where:** §3.3; Limitation 8; E12; §8.2 future
work 5.

### O12. "Why three missions? Why not one model trained on all of them?"

**Response:** The missions differ in orbit (sun-synchronous vs. precessing), radar geometry (nadir
line vs. scanning swath), and passive information (cloud-only vs. microwave). One pooled model would
have to drop to the features they share and would hide exactly these differences. Each mission is
therefore a separate case with its own dataset, training, and validation; transfer between missions
is measured separately (E12). If a reviewer prefers depth over breadth: EarthCARE is the primary case
and is reported in full, and the other two use the identical method. **Where:** §1.2, §5.9, E12,
§8.3, Limitation 7.

### O13. "GMI already sees rain. Isn't the GPM gate trivial?"

**Response:** It would be if the target were rain detection, which is why the GPM label is the
radar's added value beyond GMI (vertical structure, light rain and snow, or the GMI retrieval's
uncertainty), not rain occurrence. Combined DPR + GMI products contain the withheld radar
measurement and are never used as inputs. **Where:** §5.8, §3.8.

### O14. "TRMM only covers the tropics. Is the comparison fair?"

**Response:** Cross-mission comparisons are restricted to shared latitudes or reported by latitude
band, and TRMM's results are described as tropical and subtropical only. TRMM is included because it
flew during G5NR's own period and its visible/infrared scanner matches EarthCARE's imager closely,
which isolates the effect of the orbit. **Where:** §3.7, E12, Limitation 9.

### O15. "You train on summer and test on winter. Isn't the test unfair?"

**Response:** Partly, and it is measured. The orbit is global, so June–September already holds
southern-hemisphere winter, and most inputs describe cloud physics that does not change with the
season. Leave-one-month-out puts the other winter months into training: Nov–Jan AUC rises by 0.006
(logistic regression), 0.012 (LightGBM) and 0.014 (MLP), an upper bound because those folds also
train on more months. The ranking P6 ≥ P7 > P5 holds in every held-out month, so the time-blocked
numbers are a conservative, not a distorted, estimate. **Where:** §7 E2, §15.8, Fig. S3.
