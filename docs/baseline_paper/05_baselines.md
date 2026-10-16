# 5. Gating Policies and Evaluation

## 5.1 Task

At each decision *t* with imager-visible features *x_t*, a policy outputs fire ∈ {0, 1}. Over an
evaluation period the policy may fire on at most a fraction *d* of decisions (the duty-cycle
budget). Each decision is scored independently (contextual bandit); sequential effects are left to
the RL follow-up.

### Formal statement (LaTeX-ready)

Let $\mathcal{D} = \{(x_t, y_t, r_t)\}_{t=1}^{T}$ be the decisions in an evaluation block, with
always-available features $x_t \in \mathbb{R}^{12}$, binary label
$y_t = \mathbb{1}[r_t \ge \tau]$, nadir precipitation rate $r_t$ (`prectot_nadir`), and threshold
$\tau = 1\,\mathrm{mm\,h^{-1}}$. A gating policy $\pi: \mathbb{R}^{12} \to \{0,1\}$ must satisfy
the budget

$$\frac{1}{T}\sum_{t=1}^{T} \pi(x_t) \le d .$$

For a scoring model $s(x)$, the gate is $\pi_d(x) = \mathbb{1}[s(x) \ge \theta_d]$, where
$\theta_d$ is the $(1-d)$ quantile of $s$ on the validation block. The metrics are

$$\mathrm{capture}@d = \frac{\sum_t \pi_d(x_t)\, y_t}{\sum_t y_t}, \qquad
\mathrm{mass\text{-}capture}@d = \frac{\sum_t \pi_d(x_t)\, r_t}{\sum_t r_t},$$

$$\mathrm{precision}@d = \frac{\sum_t \pi_d(x_t)\, y_t}{\sum_t \pi_d(x_t)}, \qquad
\mathrm{lift}@d = \frac{\mathrm{capture}@d}{d}.$$

Reference values: random gating gives $\mathrm{capture}@d = d$; the oracle gives
$\min(1, d/\bar{y})$ with base rate $\bar{y} = 0.1297$.

Because decisions are scored independently and $\pi$ does not affect future observations, this is
a contextual bandit with a budget, and the budget-optimal static policy is a threshold on
$P(y=1 \mid x)$. A sequential (RL) formulation becomes necessary only when the action changes
future state (battery, switching cost) or future observations (`cpr_conditional` features).

## 5.2 Labels

- **Primary:** *y_t* = 1 if `prectot_nadir` ≥ 1 mm/hr (base rate 12.97%).
- **Secondary:** convective, `preccon_nadir` ≥ 1 mm/hr (base rate 1.03%).
- **Sensitivity:** thresholds 0.1 and 5 mm/hr, and the median-split `fire` label used by the
  current notebook/CZML export (~0.09 mm/hr, ~50% positive).

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
| ROC-AUC, AP | threshold-free ranking quality | comparability with the existing 0.837 / 0.368 numbers |

Reference curves:
- Random gating: capture@d = *d*.
- Oracle: capture@d = min(1, *d* / base rate). With base rate 12.97%, the oracle captures
  everything at *d* ≈ 13%.

Report capture@d as a curve over *d* ∈ [0, 1] and as a table at *d* ∈ {5, 10, 13, 20, 30, 50}%.

## 5.4 Policies

| # | policy | inputs | how the gate is set |
|---|---|---|---|
| P0 | always-on | none | fires every decision (*d* = 100%); upper bound without a budget |
| P1 | random | none | fires with probability *d* |
| P2 | geometry-only | Tier C features | logistic regression on geometry; thresholded to *d* |
| P3 | latitude band | `lat` | fires inside the band(s) with the highest training-set positive rate, widened until the budget is met |
| P4 | single threshold | `tautot_mean` (night: `cldtmp_mean`) | fires above the quantile that meets *d* |
| P5 | logistic regression | Tier A (and B, C) | standardized features; thresholded to *d* |
| P6 | gradient-boosted trees | Tier A (and B, C) | e.g., LightGBM / sklearn HistGradientBoosting; thresholded to *d* |
| P7 | small MLP (optional) | Tier A | 12-64-64-1; included only to cost the "neural policy" option in Section 6 |
| P8 | oracle | true label | fires on positives first |

**Threshold rule (all learned policies):** pick the score threshold that fires on exactly a
fraction *d* of the **validation** block, then apply that threshold unchanged to the test block.
Report the realized test duty cycle next to capture, since it will drift slightly.

## 5.5 Missing values

- Drop decisions whose target is invalid (`*_valid` flags) from scoring and report the count.
- For features: gradient boosting handles NaN natively; for logistic regression, impute with the
  training-block median and add a missing-indicator column. Report how many rows were affected.
- `lwtup_fore` is NaN at weather-step boundaries (~11.7k rows); keep it, with the indicator.
- `tautot` at night: see Section 6. Do not silently feed nighttime optical depth to a policy that
  claims to be onboard-realistic.

## 5.6 Preliminary results (random 30% split — superseded once E1 runs)

From [`../trackwise_method.md`](../trackwise_method.md) §4.3, against `prectot_nadir` ≥ 1 mm/hr:
`tautot_mean` AUC 0.815, `tautot_max` 0.789, `cldtot_mean` 0.651, `cldtmp_mean` 0.622 (inverted),
`lwtup_mean` 0.598 (inverted), `ocean_frac_nadir` 0.572; logistic regression on all 12 features
AUC 0.837, AP 0.368 (base rate 13.0%). These show the signal exists. They are not paper results
until re-run under the time-blocked split.

## 5.7 What counts as a positive result

The learned policies are only worth reporting as a contribution if:
- they beat P2/P3 (geometry) clearly, showing the imager adds value beyond orbit position, and
- the gap to P8 (oracle) is reported honestly, since that gap is the room left for RL.
