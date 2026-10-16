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

**Response:** Two costs must be separated. The trained policy costs ~12 to ~5,000 operations per
decision once a minute — negligible even for radiation-hardened processors, and onboard neural
inference has already flown [giuffrida2022phisat]. The real constraint is producing the input
features; Tier B/C measure how much skill survives without onboard retrievals. Training happens on
the ground. **Where:** §6, Table 5.

### O4. "The features are model diagnostics, not MSI retrievals."

**Response:** Correct and stated explicitly. They are idealized proxies; results are an upper bound
on what a real imager provides. Tier B and the night analysis bound the effect, and validation on
real EarthCARE MSI/CPR data is the first follow-up. **Where:** §3.6, Limitation 1, §8.2.

### O5. "Your AUC is inflated by temporal leakage."

**Response:** All headline results use a time-blocked split with a buffer of at least one weather
step, and the random-split numbers are shown only for comparison. **Where:** §7 E1. (This objection
is valid against the current 0.837 number — do not report it without the blocked-split
counterpart.)

### O6. "One year of one model; does it generalise?"

**Response:** Leave-one-month-out results across eight months spanning boreal summer to winter;
known G5NR biases (split ITCZ, weaker Hadley circulation [gelaro2015g5nr]) are discussed. Real-data
validation is future work. **Where:** E2, §10.5, Limitation 3.

### O7. "Why not just train RL now?"

**Response:** Without a fixed metric, split, and static baseline at the same budget, an RL result
cannot be interpreted. With lag-1 precipitation autocorrelation of only 0.17, a sequential policy
must earn its gain from budget dynamics or revealed features, which this benchmark makes testable.
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
the imager's value is reported as the margin over them. **Where:** §5.4, Table 4.

### O11. "A sun-synchronous orbit only sees two times of day. Doesn't that bias the results?"

**Response:** It limits what the results describe, and the paper states this. EarthCARE samples
~14:00 and ~02:00 local time, so every reported rate is conditional on precipitation at those times,
not on daily precipitation. The labels remain correct for the task, because a radar can only gate
what is beneath it when it passes over. The open question is transfer to orbits with different local
times. Because G5NR has every half hour, the fraction of daily precipitation the two windows miss
can be measured directly, which real EarthCARE data cannot do. **Where:** §3.2; Limitation 8;
§8.2 future work 5.
