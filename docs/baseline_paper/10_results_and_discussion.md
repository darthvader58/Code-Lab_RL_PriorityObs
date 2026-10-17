# 10. Results and Discussion (template)

Write this section only after `07_experiments.md` is filled in. Each subsection says what to report
and gives sentence templates for the likely outcomes. Delete the outcomes that did not happen.

Report §10.1–10.6 **per mission** (EarthCARE first; TRMM and GPM as their own subsections or as
columns in each table), then compare the missions in §10.8. Never pool the three datasets into one
result.

## 10.1 Leakage and the evaluation split (E1, E2)

Report: AUC/AP/capture@13% under random vs. time-blocked splits; leave-one-month-out mean and range.

- *If the time-blocked AUC drops noticeably:* "A random split overstates skill by [Δ] AUC because
  consecutive decisions share a 30-minute weather field; all further results use the time-blocked
  split." This is itself a methodological finding worth a sentence in the abstract.
- *If it barely changes:* "Skill is robust to the split ([Δ] AUC), consistent with nadir
  precipitation decorrelating within one decision (lag-1 ACF [TO RERUN])."
- Separate leakage from seasonal shift with E2: if June–September months score well in LOMO but
  the Nov–Jan test block does not, the drop is seasonal, not leakage.

## 10.2 Gating value under a budget (E3) — the main result

Report: Figure 1 and the capture table.

Sentence template: "At a 20% duty cycle, the best imager-driven gate captures [X]% of nadir
precipitation events ([lift]× random), compared with [Y]% for geometry-only gating; the oracle
captures 100%. The remaining gap of [100 − X] points is the headroom available to a sequential
policy."

Also discuss:
- Which simple policy is closest to the learned one. If the single `tautot_mean` threshold is within
  a few points of gradient boosting, say so plainly — it means the decision is nearly one-dimensional.
- Mass-capture vs. event-capture: does the gate preferentially catch heavy precipitation?

## 10.3 Label geometry (E4)

Report: the 2 × 2 grid.

Sentence template: "A policy trained and scored on swath truth would report [A]% capture; scored
against what the radar could actually measure, the same policy captures [B]%. The swath label
overstates delivered capture by [A − B] points."

Connect to prior work carefully: "Studies that evaluate targeting for nadir instruments against
scene-scale truth may overstate achievable capture; footprint-scale labels avoid this." Do not claim
any specific published study is wrong unless you have checked its labels.

## 10.4 Onboard feasibility (E5, E9)

Report: Figure 2, tier table, cost table.

- *If Tier B retains most of Tier A:* "Useful gating does not require onboard cloud retrievals:
  thermal-infrared-like quantities and orbit geometry retain [R]% of full-feature capture, day and
  night, with a policy of [N] operations per decision."
- *If Tier B retains little:* "Gating value depends on optical-depth information, which is
  daylight-only and requires onboard retrieval; this sets a concrete processing requirement for
  future missions."
- Either way, answer the compute objection directly: policy cost is negligible; feature cost
  decides feasibility.

## 10.5 Where the gate works and fails (E6, E11)

Report: stratified table, feature importance.

Expected discussion points (verify against results):
- Tropics vs. mid-latitudes: convective vs. frontal precipitation regimes.
- Ocean vs. land.
- Polar decisions: cold surfaces confuse infrared-based features (the same failure that broke the
  earlier clustering design).
- G5NR's split-ITCZ bias may shape where tropical skill appears; mention it when discussing the
  tropics.

## 10.6 Sensitivity (E7)

One paragraph: does the ranking of policies change with the precipitation threshold or for
convective-only targets? If convective (1.03% base rate) is much harder, say so — it motivates
better features rather than a better learner.

## 10.7 Implications for a sequential RL policy

Close the section with what RL would need to do to be worthwhile:
- beat the best static gate at the same duty cycle on the time-blocked test set;
- use something static gates cannot: battery state, switching costs, or `cpr_conditional` features
  revealed after firing;
- note that lag-1 precipitation autocorrelation was only ~0.17 for EarthCARE's nadir label in the
  earlier extraction ([TO RERUN]), so the gain from simply remembering the last minute is likely
  small — any RL gain has to come from the budget dynamics or the revealed features. Scanning radars
  see a wider area, so their labels may be more persistent; check per mission.

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
