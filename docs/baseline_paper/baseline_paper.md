# Baseline Paper — Master Outline

**Working title:** *Benchmarks for Autonomous Prioritization of Spaceborne Radar Observations:
EarthCARE, TRMM, and FY-3G*

**Research goal (the program):** Enable spacecraft to autonomously determine which Earth-science observations to prioritize as
the observed environment changes, using reinforcement learning to make those decisions onboard under
resource limits.

**Central research question (this paper):** What baseline skill can static policies achieve, on mission-specific benchmarks, in autonomously
prioritizing when to activate a spacecraft's active radar under a power budget using only
passive-sensor observations, and how do orbit, radar footprint, and passive sensing shape that
baseline?

All three missions pose it as **detection prioritization**: when will the radar measure precipitation
the passive sensors see only as cloud? They differ in what shapes the answer: when and where the orbit
samples (local time × latitude) and how much of the scene the radar measures (nadir line vs. scanned
swath).

**This paper's contribution to the goal:** Three mission benchmarks (datasets, labels, splits, metrics, and static baselines) that define the
changing environment, the prioritization decision, and the performance bar a reinforcement-learning
policy must exceed for each mission.

**Paper type:** dataset + baselines. No reinforcement learning is trained in this paper. It defines
**three benchmarks** (one per mission), each with its own dataset, label, splits, metric, and set of
baselines, and therefore the bar each of **three future RL policies** (one per mission) must beat.

**Scope (agreed with advisor, Oct 2026):** three missions are studied as **three separate application
cases**, each with its own dataset, training, and validation. They are not pooled into one model.
EarthCARE is the primary case and is done in full depth first. TRMM and FY-3G are added with the same
method if time allows; depth on one mission takes priority over shallow coverage of three. If TRMM or
FY-3G cannot be completed at the same depth, the paper falls back to EarthCARE with the other missions in
future work (`14_submission_plan.md`).

This file is the single-page summary of everything the paper will discuss. Each section is
expanded in its own file so it can be pasted into Overleaf section by section:

| file | paper section |
|---|---|
| [`00_abstract.md`](./00_abstract.md) | Title options, abstract, keywords, plain-language summary |
| [`01_introduction.md`](./01_introduction.md) | Introduction, motivation, contributions |
| [`02_related_work.md`](./02_related_work.md) | Related work (condensed review) |
| [`03_dataset.md`](./03_dataset.md) | The track-wise datasets (shared method + one subsection per mission) |
| [`04_label_finding.md`](./04_label_finding.md) | Label geometry: scoring against the radar's own footprint |
| [`05_baselines.md`](./05_baselines.md) | Gating policies and the evaluation metric, per mission |
| [`06_onboard_feasibility.md`](./06_onboard_feasibility.md) | Feature tiers and onboard cost, per mission |
| [`07_experiments.md`](./07_experiments.md) | Experiments to run before submission + result tables |
| [`08_limitations_future_work.md`](./08_limitations_future_work.md) | Limitations and the path to RL |
| [`09_references.md`](./09_references.md) | Reference list with verification status |
| [`10_results_and_discussion.md`](./10_results_and_discussion.md) | Results/discussion template with sentence frames per outcome |
| [`11_figures_and_tables.md`](./11_figures_and_tables.md) | Every figure and table, its data source, and status |
| [`12_reviewer_objections.md`](./12_reviewer_objections.md) | Anticipated objections and responses |
| [`13_key_numbers.md`](./13_key_numbers.md) | Every number in the paper with its source file |
| [`14_submission_plan.md`](./14_submission_plan.md) | Venues, staging of the three missions, Overleaf section map |
| [`15_earthcare_results.md`](./15_earthcare_results.md) | EarthCARE results report: figures, summary tables, verdict per research question |

**Reading order for writing:** this file → `13_key_numbers.md` → sections 00–08 → run
`07_experiments.md` → `10_results_and_discussion.md` → `12_reviewer_objections.md` →
`14_submission_plan.md`.

Status markers: **[TO RUN]** = experiment not yet run. **[TO BUILD]** = the dataset for that
mission does not exist yet. Do not submit with any of these markers left.

**Status (18 Oct 2026):** EarthCARE is complete: dataset built (352,800 decisions, no gaps), E1–E11
run including P7, results in `missions/earthcare/results/`, every EarthCARE number in
`13_key_numbers.md`. TRMM and the third mission are not started.

**Update (22 Oct 2026):** the EarthCARE figures are drawn (`python -m baselines.paper_figures
--mission earthcare` → `figures/earthcare/`) and collected with the result tables and a verdict per
research question in `15_earthcare_results.md`. Verdict: RQ1–RQ4 answered for EarthCARE; RQ5 waits
for TRMM.

**Update (27 Oct 2026):** TRMM is complete (dataset, E1–E11 with P7, notebook) and E12 compares
EarthCARE with TRMM (`missions/comparison/`). FY-3G, the third mission, has its code and
offline checks done (`docs/FY3G_trackwise_method.md`); its extraction has not been run.

---

## 1. Problem statement

The program's goal is autonomous, onboard prioritization of Earth-science observations in a changing
environment. This paper studies one concrete, measurable instance of it: when to activate a
spacecraft's active radar. Spaceborne radars measure precipitation and cloud structure that passive
sensors cannot, but they are far more expensive to run. If a radar could only run for a fraction *d*
of each orbit, which moments should it prioritize? The decision has to be made onboard, once per
minute, using only what the passive sensors on the same spacecraft observe at that moment, while the
weather below changes from one minute to the next.

Throughout the paper, a **gate** is the onboard prioritization policy that switches the radar on or
off.

The three missions pose this question under three different sampling geometries:

| | EarthCARE | TRMM | FY-3G |
|---|---|---|---|
| NORAD ID | 59908 | 25063 | 56232 |
| active sensor | CPR, 94 GHz cloud radar, **nadir only** | PR, 13.8 GHz, **cross-track scanning**, 247 km | PMR, Ku 13.6 + Ka 35.5 GHz, **cross-track scanning**, 303 km |
| passive context used | MSI (visible/infrared): sees **clouds** | VIRS (visible/infrared): sees **clouds** (TMI not used) | MERSI-RM (visible/infrared): sees **clouds** (MWRI-RM not used) |
| orbit | sun-synchronous, ~14:00 / ~02:00 local | 35°, precessing, all local times | 50°, precessing, all local times |
| latitude coverage | near-global (±82°) | tropics and subtropics (±35°) | tropics to mid-latitudes (±50°) |
| what it adds | all latitudes, two local times | all local times, tropics | all local times, to the mid-latitudes |
| prioritization type | **detection**: is there precipitation under the radar? | **detection**: is there precipitation in the radar swath? | **detection**: is there precipitation in the radar swath? |

## 2. Research questions

**Central question:** What baseline skill can static policies achieve, on mission-specific benchmarks, in autonomously
prioritizing when to activate a spacecraft's active radar under a power budget using only
passive-sensor observations, and how do orbit, radar footprint, and passive sensing shape that
baseline?

All three missions pose it as detection prioritization: when will the radar measure precipitation
the passive sensors see only as cloud?

RQ1–RQ4 answer it for each mission separately; RQ5 answers it across missions.

- **RQ1 — Predictability.** For each mission, how well does passive context predict precipitation in
  the radar's own footprint, when the evaluation is free of temporal leakage?
- **RQ2 — Prioritization value.** At a duty-cycle budget *d*, what fraction of radar-measurable precipitation
  does a passive-driven gate capture, compared with random gating, geometry-only gating, and the
  oracle?
- **RQ3 — Label geometry.** How much does scoring against the passive sensor's swath, instead of the
  radar's footprint, distort the positive rate and the apparent skill? (Largest for EarthCARE's nadir
  radar.)
- **RQ4 — Onboard feasibility.** How much skill survives with features cheap enough to compute
  onboard, and how much survives at night?
- **RQ5 — Mission comparison.** How do orbit sampling (sun-synchronous vs. precessing; tropical vs.
  mid-latitude reach) and radar geometry (nadir vs. scanning) change achievable gating skill?
  This compares three separately trained results; it does not train one model on all three.

## 3. Contributions

1. **Three track-wise datasets.** One-minute radar on/off decisions along each mission's real ground
   track (historical TLE sets, 1 s sampling), June 2005 – January 2006 of the GEOS-5 Nature Run
   (G5NR) at 0.0625° / 30 min, matched to G5NR by UTC time of day and day of year. EarthCARE
   **done** (352,800 decisions, no gaps); TRMM **done** (352,800 decisions); FY-3G **[TO BUILD; code
   ready, extraction pending]**.
2. **A label-geometry finding.** For EarthCARE, scoring against the imager swath instead of the nadir
   radar line inflates the positive rate 2.2× (13.7% → 30.0%), and 54.5% of swath-rainy minutes have
   no rain under the radar; for TRMM's scanning radar the inflation falls to 1.54×; FY-3G
   **[TO RUN]**.
3. **Budget-constrained baselines per mission.** Capture-vs-duty-cycle curves for random,
   geometry-only, single-threshold, logistic-regression, gradient-boosting, and oracle gates, under a
   time-blocked split. EarthCARE done: LightGBM AUC 0.893, 66.4% capture at 20% duty; TRMM done:
   LightGBM AUC 0.849, 40.7% capture (logistic regression 43.2%; oracle 48.7%); FY-3G **[TO RUN]**.
4. **Onboard feasibility per mission.** The same curves for cheap feature tiers, day and night, with
   the compute and memory cost of each policy. EarthCARE done: night-capable inputs keep 89% of
   capture; policies cost 3 kB – 1 MB and about 1 ms per decision; TRMM done; FY-3G **[TO RUN]**.
5. **A cross-mission comparison** of how orbit sampling and radar geometry shape gating skill,
   including whether a policy trained on one mission transfers to another. EarthCARE ↔ TRMM done
   (transfer loss ≤ 2.5 points); FY-3G **[TO RUN]**.

## 4. The datasets (→ `03_dataset.md`)

- **Unit:** one row per one-minute radar on/off decision; 352,800 decisions per mission over
  8 months (11,760 G5NR steps × 30), with no gaps by construction of the extractor.
- **Orbit:** historical TLE sets per mission (CelesTrak), so orbit maintenance is represented; TATC
  footprints and nadir points use the same closest-epoch TLE.
- **Time alignment:** orbit and weather start at the same UTC time of day (15:00) and the same day of
  year (1 June). TRMM flew during 2005, so its orbit also matches the G5NR year; EarthCARE and FY-3G
  fly in 2025–26.
- **Weather:** G5NR at 0.0625° (~7 km) and 30 min; window chosen as the wettest two-month start in the
  24-month record.
- **Extraction:** per-decision bounding-box fetch, reduced to summary values and discarded (avoids a
  ~10 TB full download).
- **Per mission:** the passive swath (MSI 150 km, VIRS 833 km, MERSI-RM 1000 km) defines the context
  features; the radar footprint (CPR nadir line, PR 247 km or PMR 303 km swath) defines the label.

## 5. Label geometry (→ `04_label_finding.md`)

The label must be computed from the cells the radar could actually measure. For EarthCARE that is a
narrow nadir line inside a 150 km imager swath, so a swath label rewards a policy for minutes the radar
could not have measured. On EarthCARE the nadir positive rate is 13.7% vs. 30.0% for the swath at
≥ 1 mm/hr, with 54.5% of swath positives having nothing under the radar. A policy trained on nadir
rain catches 66.4% of nadir rain but only 51.1% of swath rain at 20% duty. For TRMM and FY-3G the radar itself scans 247–303 km, so the gap between radar footprint and
passive swath is smaller: 1.54× on TRMM; FY-3G **[TO RUN]**.

## 6. The baselines (→ `05_baselines.md`)

- **Label (all three missions):** precipitation ≥ 1 mm/hr in the radar footprint (CPR nadir line,
  PR swath, PMR swath).
- **Primary metric:** *capture@d*, the fraction of positive minutes the gate fires on when it may fire
  on a fraction *d* of minutes. Also mass-capture, precision@d, ROC-AUC, average precision.
- **Policies:** random; always-on; geometry-only; latitude band; single-feature threshold; logistic
  regression; gradient-boosted trees (LightGBM); small MLP; oracle. One set per mission.
- **EarthCARE bar for RL:** LightGBM, AUC 0.893 [0.891, 0.896], capture@20% 66.4%.
- **Threshold rule:** each learned score is turned into a gate by picking the threshold that fires on
  exactly a fraction *d* of validation minutes.

## 7. Onboard feasibility (→ `06_onboard_feasibility.md`)

- Policy cost is negligible: on EarthCARE, 16 parameters (logistic regression) to 18,300 (LightGBM,
  ~3,300 comparisons), all about 1 ms per decision in Python against a 60 s interval.
- Feature cost is the same kind on every mission: MSI, VIRS and MERSI-RM need visible/infrared cloud
  products (optical depth needs daylight); the thermal channels work day and night.
- Tier C (geometry only) is the floor in every mission.

## 8. Experiments to run before submission (→ `07_experiments.md`)

E1–E11 are run **per mission**: time-blocked split, leave-one-month-out, capture@d curves, label
geometry grid, feature tiers × day/night, stratified skill, label sensitivity, missing-data
accounting, policy cost, bootstrap intervals, feature importance. **E12** compares the three missions
and tests cross-mission transfer.

## 9. Limitations (→ `08_limitations_future_work.md`)

G5NR diagnostics are idealized proxies, not instrument retrievals; ~7 km vs. sub-km to ~5 km radar
footprints; one simulated period (2005–06) from a model with a known split-ITCZ bias; the real radars
run continuously, so the budget models a future power-limited radar; decisions are scored
independently. EarthCARE samples only ~14:00 and ~02:00 local time; TRMM and FY-3G sample all local
times but only over many days. TRMM covers only the tropics and subtropics, FY-3G reaches ±50°, and
only EarthCARE samples poleward of 50°, at two local times. Passive context is visible/infrared on all
three missions; microwave-context gating needs brightness temperatures G5NR does not produce and is
future work. Policy timing was measured on a laptop, not on flight-like hardware
(Raspberry Pi–class or LEON-class timing is future work). Follow-ups: validation on real data,
hardware timing, and sequential RL per mission.

## 10. One-sentence claim of the paper

> Using only passive-sensor observations, an onboard prioritization policy can capture **66%** of radar-measurable
> precipitation on EarthCARE while the radar runs **20%** of the time (TRMM: **43%**, against an
> oracle bound of 49% for its 247 km swath; FY-3G: **[TO RUN]%**); night-capable inputs keep **89%** of that
> skill; and scoring a nadir radar against swath truth would have more than doubled the positive
> class.
