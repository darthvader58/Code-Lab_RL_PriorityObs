# Baseline Paper — Master Outline

**Working title:** *Benchmarks for Autonomous Prioritization of Spaceborne Radar Observations:
EarthCARE, TRMM, and GPM*

**Research goal (the program):** Enable spacecraft to autonomously determine which Earth-science observations to prioritize as
the observed environment changes, using reinforcement learning to make those decisions onboard under
resource limits.

**Central research question (this paper):** What baseline skill can static policies achieve, on mission-specific benchmarks, in autonomously
prioritizing when to activate a spacecraft's active radar under a power budget using only
passive-sensor observations, and how do orbit, radar footprint, and passive sensing shape that
baseline?

- **(a) Detection prioritization (EarthCARE, TRMM):** when will the radar measure precipitation the
  passive sensors see only as cloud?
- **(b) Value prioritization (GPM):** when will the radar add information beyond a passive sensor that
  already sees precipitation?

**This paper's contribution to the goal:** Three mission benchmarks (datasets, labels, splits, metrics, and static baselines) that define the
changing environment, the prioritization decision, and the performance bar a reinforcement-learning
policy must exceed for each mission.

**Paper type:** dataset + baselines. No reinforcement learning is trained in this paper. It defines
**three benchmarks** (one per mission), each with its own dataset, label, splits, metric, and set of
baselines, and therefore the bar each of **three future RL policies** (one per mission) must beat.

**Scope (agreed with advisor, Oct 2026):** three missions are studied as **three separate application
cases**, each with its own dataset, training, and validation. They are not pooled into one model.
EarthCARE is the primary case and is done in full depth first. TRMM and GPM are added with the same
method if time allows; depth on one mission takes priority over shallow coverage of three. If TRMM or
GPM cannot be completed at the same depth, the paper falls back to EarthCARE with the other missions in
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

**Reading order for writing:** this file → `13_key_numbers.md` → sections 00–08 → run
`07_experiments.md` → `10_results_and_discussion.md` → `12_reviewer_objections.md` →
`14_submission_plan.md`.

Status markers: **[TO RUN]** = experiment not yet run. **[TO RERUN]** = a number exists from the
earlier EarthCARE extraction, which paired each orbit position with weather from the wrong time of
day and season and has been deleted (`../trackwise_method.md` §7); the number must be recomputed
from the new extraction before use. **[TO BUILD]** = the dataset for that mission does not exist yet.
Do not submit with any of these markers left.

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

The three missions pose this question in three different forms:

| | EarthCARE | TRMM | GPM Core |
|---|---|---|---|
| NORAD ID | 59908 | 25063 | 39574 |
| active sensor | CPR, 94 GHz cloud radar, **nadir only** | PR, 13.8 GHz, **cross-track scanning** | DPR, Ku 13.6 + Ka 35.5 GHz, **cross-track scanning** |
| passive context | MSI (visible/infrared): sees **clouds** | VIRS (visible/infrared): sees **clouds**; TMI (microwave): sees rain | GMI (13-channel microwave): sees **rain and ice** |
| orbit | sun-synchronous, ~14:00 / ~02:00 local | 35°, precessing, all local times | 65°, precessing, all local times |
| latitude coverage | near-global | tropics and subtropics | up to ~65° |
| prioritization type | **detection**: is there precipitation under the radar? | **detection**: is there precipitation in the radar swath? | **value**: does the radar add enough beyond what GMI already sees? |

## 2. Research questions

**Central question:** What baseline skill can static policies achieve, on mission-specific benchmarks, in autonomously
prioritizing when to activate a spacecraft's active radar under a power budget using only
passive-sensor observations, and how do orbit, radar footprint, and passive sensing shape that
baseline?

- **(a) Detection prioritization (EarthCARE, TRMM):** when will the radar measure precipitation the
  passive sensors see only as cloud?
- **(b) Value prioritization (GPM):** when will the radar add information beyond a passive sensor that
  already sees precipitation?

RQ1–RQ4 answer it for each mission separately; RQ5 answers it across missions.

- **RQ1 — Predictability.** For each mission, how well does passive context predict precipitation in
  the radar's own footprint, when the evaluation is free of temporal leakage?
- **RQ2 — Prioritization value.** At a duty-cycle budget *d*, what fraction of radar-measurable precipitation
  does a passive-driven gate capture, compared with random gating, geometry-only gating, and the
  oracle? (For GPM: what fraction of the radar's added value.)
- **RQ3 — Label geometry.** How much does scoring against the passive sensor's swath, instead of the
  radar's footprint, distort the positive rate and the apparent skill? (Largest for EarthCARE's nadir
  radar.)
- **RQ4 — Onboard feasibility.** How much skill survives with features cheap enough to compute
  onboard, and how much survives at night?
- **RQ5 — Mission comparison.** How do orbit (sun-synchronous vs. precessing), radar geometry (nadir
  vs. scanning), and passive information (cloud-only vs. microwave) change achievable gating skill?
  This compares three separately trained results; it does not train one model on all three.

## 3. Contributions

1. **Three track-wise datasets.** One-minute radar on/off decisions along each mission's real ground
   track (historical TLE sets, 1 s sampling), June 2005 – January 2006 of the GEOS-5 Nature Run
   (G5NR) at 0.0625° / 30 min, matched to G5NR by UTC time of day and day of year. EarthCARE
   **[extraction in progress]**; TRMM **[TO BUILD]**; GPM **[TO BUILD; data source open, §5.8]**.
2. **A label-geometry finding.** For EarthCARE, scoring against the imager swath instead of the nadir
   radar line inflated the positive rate by about 2× in the earlier extraction **[TO RERUN]**; the
   effect for scanning radars (TRMM, GPM) is expected to be smaller and is measured **[TO RUN]**.
3. **Budget-constrained baselines per mission.** Capture-vs-duty-cycle curves for random,
   geometry-only, single-threshold, logistic-regression, gradient-boosting, and oracle gates, under a
   time-blocked split. **[TO RUN]**
4. **Onboard feasibility per mission.** The same curves for cheap feature tiers, day and night, with
   the compute and memory cost of each policy. **[TO RUN]**
5. **A cross-mission comparison** of how orbit, radar geometry, and passive information shape gating
   skill, including whether a policy trained on one mission transfers to another. **[TO RUN]**

## 4. The datasets (→ `03_dataset.md`)

- **Unit:** one row per one-minute radar on/off decision; 352,800 decisions per mission over
  8 months (11,760 G5NR steps × 30), with no gaps by construction of the extractor.
- **Orbit:** historical TLE sets per mission (CelesTrak), so orbit maintenance is represented; TATC
  footprints and nadir points use the same closest-epoch TLE.
- **Time alignment:** orbit and weather start at the same UTC time of day (15:00) and the same day of
  year (1 June). TRMM flew during 2005, so its orbit also matches the G5NR year.
- **Weather:** G5NR at 0.0625° (~7 km) and 30 min; window chosen as the wettest two-month start in the
  24-month record.
- **Extraction:** per-decision bounding-box fetch, reduced to summary values and discarded (avoids a
  ~10 TB full download).
- **Per mission:** the passive swath (MSI 150 km, VIRS ~720 km, GMI ~885 km) defines the context
  features; the radar footprint (CPR nadir line, PR or DPR ~245 km swath) defines the label.

## 5. Label geometry (→ `04_label_finding.md`)

The label must be computed from the cells the radar could actually measure. For EarthCARE that is a
narrow nadir line inside a 150 km imager swath, so a swath label rewards a policy for minutes the radar
could not have measured. The earlier EarthCARE extraction gave a nadir positive rate of ~13% vs. ~29%
for the swath at ≥ 1 mm/hr, with ~55% of swath positives having nothing under the radar
**[TO RERUN]**. For TRMM and GPM the radar itself scans ~245 km, so the gap between radar footprint and
passive swath is a different and probably smaller effect **[TO RUN]**.

## 6. The baselines (→ `05_baselines.md`)

- **EarthCARE and TRMM label:** precipitation ≥ 1 mm/hr in the radar footprint.
- **GPM label:** the radar's added value beyond GMI (candidate definitions in `05_baselines.md` §5.8),
  since GMI already detects rain.
- **Primary metric:** *capture@d*, the fraction of positive minutes the gate fires on when it may fire
  on a fraction *d* of minutes. Also mass-capture, precision@d, ROC-AUC, average precision.
- **Policies:** random; always-on; geometry-only; single-feature threshold; logistic regression;
  gradient-boosted trees; oracle. One set per mission.
- **Threshold rule:** each learned score is turned into a gate by picking the threshold that fires on
  exactly a fraction *d* of validation minutes.

## 7. Onboard feasibility (→ `06_onboard_feasibility.md`)

- Policy cost is negligible for every mission (≈ 12 to ≈ 5,000 operations per decision).
- Feature cost differs by mission: MSI and VIRS need visible/infrared cloud products (optical depth
  needs daylight); GMI brightness temperatures are available day and night.
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
independently. EarthCARE samples only ~14:00 and ~02:00 local time; TRMM and GPM sample all local
times but only over many days. TRMM covers only the tropics and subtropics. GPM's passive context
(microwave brightness temperatures) is not produced by G5NR, which forces a choice between a forward
model, proxies, or real GPM data. Follow-ups: validation on real data, and sequential RL per mission.

## 10. One-sentence claim of the paper

> Using only passive-sensor observations, an onboard prioritization policy can capture **[TO RUN]%** of radar-measurable
> precipitation on EarthCARE while the radar runs **[TO RUN]%** of the time (TRMM: **[TO RUN]%**;
> GPM: **[TO RUN]%** of the radar's added value); a cheap feature set keeps **[TO RUN]%** of that
> skill; and scoring a nadir radar against swath truth would have roughly doubled the positive class.
