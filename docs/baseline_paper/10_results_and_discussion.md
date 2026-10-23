# 10. Results and Discussion

Each subsection says what to report. **EarthCARE paragraphs are written** from the results in
`07_experiments.md` and `13_key_numbers.md`; the sentence templates remain for TRMM and GPM. Delete
the outcomes that did not happen.

EarthCARE figures and the per-question verdict are in `15_earthcare_results.md`.

Report §10.1–10.6 **per mission** (EarthCARE first; TRMM and GPM as their own subsections or as
columns in each table), then compare the missions in §10.8. Never pool the three datasets into one
result.

## 10.1 Leakage and the evaluation split (E1, E2)

Report: AUC/AP/capture@13% under random vs. time-blocked splits; leave-one-month-out mean and range.

- *If the time-blocked AUC drops noticeably:* "A random split overstates skill by [Δ] AUC because
  consecutive decisions share a 30-minute weather field; all further results use the time-blocked
  split." This is itself a methodological finding worth a sentence in the abstract.
- *If it barely changes:* "Skill is robust to the split ([Δ] AUC), consistent with nadir
  precipitation decorrelating within one decision (lag-1 ACF [value for the mission])."
- Separate leakage from seasonal shift with E2: if June–September months score well in LOMO but
  the Nov–Jan test block does not, the drop is seasonal, not leakage.

**EarthCARE (written):** "Skill is robust to the split: logistic regression scores AUC 0.833 on a
random 30% split and 0.837 on the time-blocked split, consistent with nadir precipitation
decorrelating within one decision (lag-1 autocorrelation 0.106). Holding out each month in turn gives
AUC 0.811–0.844 for logistic regression and 0.887–0.907 for gradient boosting, so the skill is not
tied to one season. Training on summer only does cost the flexible models a little on winter:
with the other winter months in training, Nov–Jan AUC rises by 0.006 (LR), 0.012 (LightGBM) and
0.014 (MLP), an upper bound since those folds also train on 7 months instead of 4. The ranking is
the same in every month, so the time-blocked numbers are conservative. All further results use the
time-blocked split."

## 10.2 Gating value under a budget (E3) — the main result

Report: Fig. 3 and the capture table (Table 4).

Sentence template: "At a 20% duty cycle, the best imager-driven gate captures [X]% of nadir
precipitation events ([lift]× random), compared with [Y]% for geometry-only gating; the oracle
captures 100%. The remaining gap of [100 − X] points is the headroom available to a sequential
policy."

Also discuss:
- Which simple policy is closest to the learned one. If the single `tautot_mean` threshold is within
  a few points of gradient boosting, say so plainly — it means the decision is nearly one-dimensional.
- Mass-capture vs. event-capture: does the gate preferentially catch heavy precipitation?

**EarthCARE (written):** "At a 20% duty cycle, the best passive-driven gate (LightGBM) captures 66.4%
of nadir precipitation events (3.4× random; AUC 0.893 [0.891, 0.896]), compared with 27.5% for
geometry-only gating and 28.0% for a latitude-band rule; the oracle captures 100% because only 12.8%
of test minutes are rainy. The remaining gap of 33.6 points is the headroom available to a
sequential policy. The decision is not one-dimensional: the best single-feature rule reaches 41.8%,
24.6 points below LightGBM. Optical depth alone ranks rain about as well as logistic regression on
all features (AUC 0.834 vs. 0.837), and the remaining gain comes from non-linear models: a small MLP
reaches 65.1% (AUC 0.889), within 1.4 points of LightGBM, though LightGBM is ahead in every bootstrap
resample. The gate catches 72.7% of the precipitation amount, more than its 66.4% of events, so it
favours heavier precipitation; the oracle, which fires on every rainy minute regardless of
intensity, reaches 91.7% of the amount."

## 10.3 Label geometry (E4)

Report: the 2 × 2 grid.

EarthCARE result (filled): "Scoring against the imager swath more than doubles the positive class
(13.7% → 30.0% of minutes) and inflates precision at a 20% budget from 44.7% to 75.0%: a
swath-scored study would claim the radar finds rain on most of its on-time when it does so on fewer
than half. For convective precipitation the inflation is larger: 77% of swath-convective minutes have
none at nadir, and precision rises from 5.9% to 25.5%.
Under the same budget, capture moves the other way (51% on the swath label vs. 66% on the nadir
label), because the larger swath class cannot be covered by 20% of minutes. Ranking quality is
unaffected (AUC 0.89–0.90 in all four train/score combinations), so the label choice changes the
reported numbers, not which scenes the policy prefers."

Connect to prior work carefully: "Studies that evaluate targeting for nadir instruments against
scene-scale truth may overstate achievable capture; footprint-scale labels avoid this." Do not claim
any specific published study is wrong unless you have checked its labels.

## 10.4 Onboard feasibility (E5, E9)

Report: Fig. 5, tier and cost table (Table 5).

- *If Tier B retains most of Tier A:* "Useful gating does not require onboard cloud retrievals:
  thermal-infrared-like quantities and orbit geometry retain [R]% of full-feature capture, day and
  night, with a policy of [N] operations per decision."
- *If Tier B retains little:* "Gating value depends on optical-depth information, which is
  daylight-only and requires onboard retrieval; this sets a concrete processing requirement for
  future missions."
- Either way, answer the compute objection directly: policy cost is negligible; feature cost
  decides feasibility.

**EarthCARE (written):** "Useful gating does not require onboard cloud retrievals: thermal-infrared-like
quantities and orbit geometry (Tier B) retain 90% of full-feature capture when the gate switches
between day and night inputs, and at night they match the richer feature set exactly (49.8% at 20%
duty). Optical depth adds about 12 points by day. Geometry alone retains 57%. The policies cost
16 to 18,300 parameters (3 kB to 1 MB) and about 1 ms per decision in Python, against a 60-second
decision interval, so policy cost is negligible and feature cost decides feasibility."

## 10.5 Where the gate works and fails (E6, E11)

Report: stratified table, feature importance.

Expected discussion points (verify against results):
- Tropics vs. mid-latitudes: convective vs. frontal precipitation regimes.
- Ocean vs. land.
- Polar decisions: cold surfaces confuse infrared-based features (the same failure that broke the
  earlier clustering design).
- G5NR's split-ITCZ bias may shape where tropical skill appears; mention it when discussing the
  tropics.

**EarthCARE (written):** "Ranking is weakest in the tropics (AUC 0.846), where precipitation is most
frequent (19.3% of minutes), and strongest over land (0.950) and poleward of 60° (0.940), where it is
rarest (6.7% and 4.4%). With one global threshold, the gate spends 30.2% duty in the tropics and
6.2% in polar regions, so the budget follows where rain is frequent; polar capture (54.6%) is the
lowest of any region despite its high AUC, which a per-region threshold could improve.
Optical depth dominates the model (permutation AUC drop 0.207 for `tautot_max`, 0.091 for
`tautot_mean`), followed by outgoing longwave (0.052) and cloud fraction (0.038)."

## 10.6 Sensitivity (E7)

One paragraph: does the ranking of policies change with the precipitation threshold or for
convective-only targets? If convective (1.2% base rate) is much harder, say so — it motivates
better features rather than a better learner.

**EarthCARE (written):** "LightGBM beats logistic regression on all five labels (AUC 0.893–0.957),
so the ranking of policies does not depend on the threshold. Rare targets are ranked well but are
hard to catch within a matching budget: with the radar on as often as the event occurs, the gate
catches 41.1% of ≥ 5 mm/hr minutes (3.6% of minutes) and 30.4% of convective minutes (1.2%), against
54.9% at 1 mm/hr."

## 10.7 Implications for a sequential RL policy

Close the section with what RL would need to do to be worthwhile:
- beat the best static gate at the same duty cycle on the time-blocked test set;
- use something static gates cannot: battery state, switching costs, or `cpr_conditional` features
  revealed after firing;
- note that lag-1 precipitation autocorrelation is only 0.106 for EarthCARE's nadir label, so the
  gain from simply remembering the last minute is likely small — any RL gain has to come from the
  budget dynamics or the revealed features. The observed features persist longer (lag-1 0.64 for
  optical depth, 0.84 for longwave), so recent observations can still carry information. Scanning
  radars see a wider area, so their labels may be more persistent; check per mission.
- **EarthCARE bar:** at the same duty cycle on the Nov–Jan test block, beat LightGBM's capture (66.4%
  at 20%, 50.8% at 13%; AUC 0.893). A policy that adapts its duty cycle to the scene could also be
  judged on capture per unit of radar on-time over the whole period.

## 10.8 Comparing the missions (E12)

Report: the side-by-side comparison table and the transfer matrix.

Discuss each difference through the property that causes it:
- **Orbit / local time.** "On TRMM, decisions near 14:00 and 02:00 local time give capture@20% of
  [X]%, against [Y]% at other hours; EarthCARE's two sampling times therefore [over/under]state
  gating skill relative to the full day."
- **Radar geometry.** "The scanning radars' base rates are [higher], and the swath-label inflation
  falls from [~2×] (EarthCARE, nadir) to [Z×] (TRMM, GPM)."
- **Passive information.** "With microwave context, the gate [does/does not] find where the radar
  adds value beyond GMI: capture of added value at 20% is [V]%."
- **Transfer.** "A policy trained on EarthCARE loses [L] points of capture when applied to TRMM,
  compared with TRMM's own policy." A large loss supports training one policy per mission.

State plainly when the missions disagree; three separate cases may give three answers, and that is a
result, not a failure.
