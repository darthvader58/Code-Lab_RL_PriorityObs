# 5. Gating Policies and Evaluation

## 5.1 Task

At each decision *t* with passive-sensor features *x_t*, a policy outputs fire ∈ {0, 1}. Over an
evaluation period the policy may fire on at most a fraction *d* of decisions (the duty-cycle
budget). Each decision is scored independently (contextual bandit); sequential effects are left to
the RL follow-up.

The gate is an onboard observation-prioritization policy. On all three missions it performs
**detection prioritization**: find precipitation the passive sensors see only as cloud.

The task is defined and solved **separately for each mission**: each has its own features, label,
splits, and fitted policies. The formal statement below is written for EarthCARE; TRMM uses the same
form with the PR-swath label, and FY-3G with the PMR-swath label (§5.8).

### Formal statement (LaTeX-ready)

Let $\mathcal{D} = \{(x_t, y_t, r_t)\}_{t=1}^{T}$ be the decisions in an evaluation block, with
always-available features $x_t \in \mathbb{R}^{13}$ (Tier A, §6.3), binary label
$y_t = \mathbb{1}[r_t \ge \tau]$, nadir precipitation rate $r_t$ (`prectot_nadir`), and threshold
$\tau = 1\,\mathrm{mm\,h^{-1}}$. A gating policy $\pi: \mathbb{R}^{13} \to \{0,1\}$ must satisfy
the budget

$$\frac{1}{T}\sum_{t=1}^{T} \pi(x_t) \le d .$$

For a scoring model $s(x)$, the gate is $\pi_d(x) = \mathbb{1}[s(x) \ge \theta_d]$, where
$\theta_d$ is the $(1-d)$ quantile of $s$ on the validation block. The metrics are

$$\mathrm{capture}@d = \frac{\sum_t \pi_d(x_t)\, y_t}{\sum_t y_t}, \qquad
\mathrm{mass\text{-}capture}@d = \frac{\sum_t \pi_d(x_t)\, r_t}{\sum_t r_t},$$

$$\mathrm{precision}@d = \frac{\sum_t \pi_d(x_t)\, y_t}{\sum_t \pi_d(x_t)}, \qquad
\mathrm{lift}@d = \frac{\mathrm{capture}@d}{d}.$$

Reference values: random gating gives $\mathrm{capture}@d = d$; the oracle gives
$\min(1, d/\bar{y})$ with base rate $\bar{y}$ (0.128 on the EarthCARE test block).

Because decisions are scored independently and $\pi$ does not affect future observations, this is
a contextual bandit with a budget, and the budget-optimal static policy is a threshold on
$P(y=1 \mid x)$. A sequential (RL) formulation becomes necessary only when the action changes
future state (battery, switching cost) or future observations (`cpr_conditional` features).

## 5.2 Labels

- **EarthCARE primary:** *y_t* = 1 if `prectot_nadir` ≥ 1 mm/hr (base rate 13.7% overall, 12.8% in
  the test block).
- **TRMM primary:** *y_t* = 1 if `prectot_radar` (maximum `prectot` in the PR swath) ≥ 1 mm/hr;
  mass-capture is weighted by `prectot_radar_mean`, the mean over the PR swath (rain area × rate)
  (base rate 41.6% overall).
- **FY-3G primary:** *y_t* = 1 if `prectot_radar` (maximum `prectot` in the PMR swath) ≥ 1 mm/hr,
  weighted the same way (§5.8) **[dataset pending]**.
- **Secondary (all):** convective, `preccon` ≥ 1 mm/hr in the radar footprint (EarthCARE 1.2%).
- **Sensitivity:** thresholds 0.1 and 5 mm/hr, and the median-split `fire` label used by the
  CZML export (0.097 mm/hr on EarthCARE, ~47% positive in the test block).

Why 1 mm/hr is primary: it is a physically meaningful rain-rate threshold and gives a realistic
minority class. A median split is balanced by construction, which hides the base-rate problem the
budget metric is meant to expose.

## 5.3 Metrics

| metric | definition | why |
|---|---|---|
| **capture@d** (primary) | fraction of positive decisions on which the policy fires, when it fires on a fraction *d* of decisions | directly answers "how much precipitation does the radar get for its budget?" |
| **mass-capture@d** | fraction of total nadir precipitation (sum of `prectot_nadir`) in fired decisions | rewards catching heavy events, not just any event |
| **precision@d** | fraction of fired decisions that are positive | how much of the radar's on-time is wasted |
| **lift@d** | capture@d ÷ *d* | improvement over random gating (random = 1.0) |
| ROC-AUC, AP | threshold-free ranking quality | comparability with earlier numbers and other studies |

Reference curves:
- Random gating: capture@d = *d*.
- Oracle: capture@d = min(1, *d* / base rate). With a base rate near 13%, the oracle captures
  everything at *d* ≈ 13%. The scanning radars see a much wider area, so their base rates will be
  higher and their oracle curves saturate later.

Report capture@d as a curve over *d* ∈ [0, 1] and as a table at *d* ∈ {5, 10, 13, 20, 30, 50}%.

## 5.4 Policies

| # | policy | inputs | how the gate is set |
|---|---|---|---|
| P0 | always-on | none | fires every decision (*d* = 100%); the right end of every capture curve, so not run separately |
| P1 | random | none | fires with probability *d* |
| P2 | geometry-only | Tier C features | logistic regression on geometry; thresholded to *d* |
| P3 | latitude band | `lat` | score = training-set positive rate of the decision's 5° latitude band; thresholded to *d* |
| P4 | single threshold | `tautot_mean` (night: `cldtmp_mean`) | fires above the quantile that meets *d* |
| P5 | logistic regression | Tier A (and B, C) | standardized features; thresholded to *d* |
| P6 | gradient-boosted trees | Tier A (and B, C) | LightGBM, 300 trees; NaN handled natively; thresholded to *d* |
| P7 | small MLP | Tier A | 15-64-64-1 (13 features + 2 missing-value indicators), early stopping; costs the "neural policy" option in Section 6 |
| P8 | oracle | true label | fires on positives first |

**Threshold rule (all learned policies):** pick the score threshold that fires on exactly a
fraction *d* of the **validation** block, then apply that threshold unchanged to the test block.
Report the realized test duty cycle next to capture, since it will drift slightly.

## 5.5 Missing values

- Drop decisions whose target is invalid (`*_valid` flags) from scoring and report the count.
- For features: gradient boosting handles NaN natively; for logistic regression, impute with the
  training-block median and add a missing-indicator column. Report how many rows were affected.
- `lwtup_fore` is NaN at weather-step boundaries (11,760 rows on EarthCARE); keep it, with the
  indicator. `cldtmp_mean` is NaN in 17,425 rows (no cloud top in the swath).
- `tautot` at night: see Section 6. Do not silently feed nighttime optical depth to a policy that
  claims to be onboard-realistic.

## 5.6 EarthCARE results (time-blocked test, Nov 2005 – Jan 2006)

Full tables in `13_key_numbers.md` and `missions/earthcare/results/`. At 20% duty, capture of rainy
minutes: random 19.7%, geometry LR 27.5%, latitude band 28.0%, single threshold 41.8%, logistic
regression 60.3%, MLP 65.1%, **LightGBM 66.4% (AUC 0.893 [0.891, 0.896], AP 0.544)**, oracle 100%.
Single features (E11): `tautot_mean` AUC 0.834, `tautot_max` 0.812, `cldtot_mean` 0.652,
`cldtmp_mean` 0.619 (inverted). Both §5.7 conditions are met: the learned policies beat geometry by
33–39 capture points, and the gap to the oracle at 20% duty (33.6 points) is the room left for RL.

**The EarthCARE bar for an RL policy:** at the same duty cycle on the same test block, catch more
rainy minutes than LightGBM (66.4% at 20%, 50.8% at 13%), or rank them better (AUC > 0.893).

## 5.7 What counts as a positive result

The learned policies are only worth reporting as a contribution if:
- they beat P2/P3 (geometry) clearly, showing the imager adds value beyond orbit position, and
- the gap to P8 (oracle) is reported honestly, since that gap is the room left for RL.

## 5.8 FY-3G: the same detection label on a third orbit

FY-3G uses TRMM's label form: *y_t* = 1 if `prectot_radar` (maximum `prectot` in the 303 km PMR
swath) ≥ 1 mm/hr, with mass-capture weighted by `prectot_radar_mean`. The threshold stays 1 mm/hr on
all three missions so they remain comparable, although the PMR's Ka band detects lighter rain than
TRMM's PR; a lower threshold is part of E7, not the primary label. Feature tiers are TRMM's
(`06_onboard_feasibility.md` §6.7), since MERSI-RM-like and VIRS-like inputs are the same G5NR
proxies. E6 uses bands < 23.5°, 23.5–35° and ≥ 35°; the last is the latitude range FY-3G adds over
TRMM.

**Why there is no value-prioritization label.** All three gates see clouds through visible/infrared
channels, so detecting rain is the right objective. A gate fed by a microwave imager, which already
sees rain, would instead have to score what the radar adds beyond it (vertical structure, light rain
and snow, or the passive retrieval's uncertainty); that case is future work
(`08_limitations_future_work.md`).

## 5.9 One policy set per mission; cross-mission transfer

The policies P0–P8 are fitted and evaluated separately for each mission. A single model trained on
all three missions is not used: their label footprints (nadir line vs. 247 or 303 km swath), base
rates, latitude ranges and feature sets differ, and pooling would reduce them to their shared
features. Transfer is
measured separately in E12, after each mission's own results exist: train on one mission's data,
apply to another's with the shared features (geometry tier, cloud proxies), and report the loss
relative to that mission's own policy. A large loss when transferring from EarthCARE to TRMM, for
example, would show that a policy trained only on ~14:00/~02:00 weather does not cover other times of
day.
