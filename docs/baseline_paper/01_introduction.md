# 1. Introduction

## 1.1 Motivation

Earth-observing spacecraft increasingly need to decide for themselves which observations to make.
Conditions below them change faster than a ground-planned schedule can follow, and their most
capable instruments are often their most resource-expensive. The long-term goal of this work is:

> Enable spacecraft to autonomously determine which Earth-science observations to prioritize as
> the observed environment changes, using reinforcement learning to make those decisions onboard under
> resource limits.

Reinforcement learning is a natural tool for such decisions, but it needs benchmarks: a defined
environment, a decision, a measure of success, and a baseline to beat. This paper provides them for
one concrete instance of the problem — when to activate a spacecraft's active radar.

Spaceborne cloud and precipitation radars measure what passive sensors cannot: the vertical
structure of cloud and precipitation. That capability is expensive. EarthCARE's Cloud Profiling Radar
(CPR) draws roughly 308 W, comparable to the ATLID lidar (310 W), while its Multi-Spectral Imager
(MSI) is a low-power passive instrument imaging a 150 km swath continuously.

The radars on EarthCARE, TRMM, and FY-3G all run continuously. The question this paper asks is aimed at
the next generation of radar missions — small satellites and constellations where power, thermal, or
downlink budgets may not allow a radar to run all the time. On such a mission, the radar can only be
switched on for a fraction *d* of each orbit, and the choice of *which* minutes matters: a radar
running over clear sky collects nothing a passive sensor could not have reported.

Three flown missions carry a radar beside a visible/infrared imager on one spacecraft, and they differ
in exactly the ways that should change how well a passive-informed gate can work:

- **EarthCARE** (2024–): nadir-only CPR beside a visible/infrared imager (MSI), in a sun-synchronous
  orbit that always passes at ~14:00 or ~02:00 local time.
- **TRMM** (1997–2015): a cross-track scanning precipitation radar (PR) beside a visible/infrared
  scanner (VIRS) and a microwave imager (TMI), in a 35° precessing orbit that samples all local times
  over the tropics and subtropics.
- **FY-3G** (2023–): a dual-frequency (Ku/Ka) cross-track scanning precipitation measurement radar
  (PMR) beside a visible/infrared imager (MERSI-RM) and a microwave imager (MWRI-RM), in a 50°
  precessing orbit that samples all local times from the tropics into the mid-latitudes.

Together they sample every local time up to ±50° latitude, and two local times to ±82°. Studying all
three separates two effects: when in the day and where a mission can look (orbit), and how much the
radar sees compared with the passive sensor (nadir line vs. scanning swath). The passive context is
held fixed in kind (visible/infrared on all three; the microwave imagers are not used), so the
comparison isolates the orbit and the radar. The central question of this paper is therefore:

> What baseline skill can static policies achieve, on mission-specific benchmarks, in autonomously
> prioritizing when to activate a spacecraft's active radar under a power budget using only
> passive-sensor observations, and how do orbit, radar footprint, and passive sensing shape that
> baseline?

On all three missions this is **detection prioritization**: when will the radar measure precipitation
the passive sensors see only as cloud?

Each mission gets its own benchmark (dataset, label, splits, metric, baselines), and later its own RL
policy; the comparison across the three answers the question.

## 1.2 The decision problem

Once per minute along the orbit, an onboard prioritization policy (the **gate**) decides whether to
power the radar for the next minute. It may use only what is available at decision time — the passive sensors' view of the
scene and the spacecraft's own geometry. It is scored on what the radar would have measured in **its
own footprint**: the nadir line for EarthCARE, the scanned swath for TRMM and FY-3G. This is
**detection prioritization**: the passive sensors see clouds, and the gate must find precipitation for
the radar to measure. A mission whose passive sensor already sees rain (a microwave imager) would
pose a different question, value prioritization, which is left to future work (§8).

Each mission is a separate application case with its own dataset, training, and validation, and can
give a different answer. One model trained on all three would have to reduce them to the features
they share and would blur exactly the differences this paper measures; cross-mission transfer is
tested separately (E12) after each mission is done.

This is the simplest form of the problem: each decision is scored independently (a contextual
bandit). Sequential structure — battery state, and radar-only features that become visible after
firing — is deliberately left out of this paper (see `08_limitations_future_work.md`) so that the
baselines here become the reference point later RL policies must beat, one per mission.

## 1.3 Why a baseline paper first

- The program's goal is RL-based prioritization; RL needs a defined environment and a bar to beat.
  These benchmarks are that environment and bar, one per mission.
- A learned sequential policy is only interesting if it beats a well-tuned static classifier
  thresholded to the same budget. That bar does not exist yet for any of the three missions.
- The evaluation choices — label geometry, train/test split, metric — change the conclusions more
  than the choice of learner. They should be fixed and published before any RL result.
- The onboard-compute objection ("a satellite cannot run an RL agent") is answerable with
  measurements: policy cost and the skill retained by cheap features.

## 1.4 Contributions

1. **Three track-wise benchmarks for autonomous observation prioritization** of one-minute radar on/off decisions along the real ground tracks
   of EarthCARE, TRMM, and FY-3G (historical orbital elements), over 8 months of the 7 km GEOS-5 Nature
   Run, matched by UTC time of day and day of year, each with radar-footprint and passive-swath truth,
   feature tiers, validity flags, and a published hash.
2. **A label-geometry finding:** scoring a nadir radar against imager-swath truth roughly doubles the
   positive class; the size of the same effect for scanning radars is measured.
3. **Budget-constrained baselines per mission**, reported as capture-vs-duty-cycle curves under a
   leakage-free, time-blocked split.
4. **An onboard-feasibility analysis per mission** comparing full, cheap, and geometry-only feature
   tiers, day and night, with per-decision compute and memory cost.
5. **A cross-mission comparison** showing how orbit local-time and latitude sampling and radar
   geometry shape gating skill, including a cross-mission transfer test.

## 1.5 Paper roadmap

Section 2 reviews onboard autonomous targeting and dynamic targeting. Section 3 describes the shared
method and the three datasets. Section 4 presents the label-geometry finding. Section 5 defines the
gating policies and the metric. Section 6 analyses onboard feasibility. Section 7 reports results per
mission and compares the missions. Section 8 discusses limitations and the path to sequential RL.

---

### Writing notes (delete before submission)

- Keep the radars-run-continuously caveat in §1.1. Reviewers will raise it if you don't.
- Name one concrete power-limited target mission in §1.1. Best candidate: INCUS — three small
  satellites, each with a RainCube-heritage Ka-band scanning radar, launch listed for 2027
  [tanelli2023incus]; RainCube itself flew a radar on a 6U CubeSat [peral2019raincube]. INCUS's
  scanning Ka-band radars make the TRMM and FY-3G cases directly relevant (FY-3G's PMR has a Ka
  channel). Quote a power or duty-cycle
  figure only after reading it in the mission paper.
- If the paper ends up EarthCARE-only (see `14_submission_plan.md`), cut §1.1's three-mission list to
  one sentence pointing to future work, and drop contribution 5.
