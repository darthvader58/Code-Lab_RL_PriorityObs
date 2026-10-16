# 1. Introduction

## 1.1 Motivation

Spaceborne cloud and precipitation radars measure what passive imagers cannot: the vertical
structure of cloud and precipitation directly beneath the spacecraft. That capability is expensive.
EarthCARE's Cloud Profiling Radar (CPR) draws roughly 308 W, comparable to the ATLID lidar (310 W),
while the Multi-Spectral Imager (MSI) is a low-power passive instrument that images a 150 km swath
continuously.

EarthCARE itself operates the CPR continuously. The question this paper asks is aimed at the next
generation of radar missions — small satellites and constellations where power, thermal, or
downlink budgets may not allow a radar to run all the time. On such a mission, the radar can only be
switched on for a fraction *d* of each orbit, and the choice of *which* minutes matters: a radar
running over clear sky collects nothing a passive imager could not have reported.

## 1.2 The decision problem

Once per minute along the orbit, an onboard controller decides whether to power the radar for the
next minute. It may use only what is available at decision time — the imager's view of the scene and
the spacecraft's own geometry. It is scored on whether the radar would have measured precipitation
in its **nadir footprint**, not anywhere in the imager's swath.

This is the simplest form of the problem: each decision is scored independently (a contextual
bandit). Sequential structure — battery state, and radar-only features that become visible after
firing — is deliberately left out of this paper (see `08_limitations_future_work.md`) so that the
baselines here become the reference point a later RL policy must beat.

## 1.3 Why a baseline paper first

- A learned sequential policy is only interesting if it beats a well-tuned static classifier
  thresholded to the same budget. That bar does not exist yet.
- The evaluation choices — label geometry, train/test split, metric — change the conclusions more
  than the choice of learner. They should be fixed and published before any RL result.
- The onboard-compute objection ("a satellite cannot run an RL agent") is answerable with
  measurements: policy cost and the skill retained by cheap features.

## 1.4 Contributions

1. **A track-wise benchmark** of 351,639 one-minute radar on/off decisions along EarthCARE's real
   ground track over 8 months of the 7 km GEOS-5 Nature Run, with nadir and swath truth, feature
   tiers, validity flags, and a published hash.
2. **A label-geometry finding:** scoring a nadir radar against swath-scale truth more than doubles
   the positive class (28.9% vs. 13.0%), and over half of swath-positive minutes have nothing for the
   radar to measure.
3. **Budget-constrained baselines** reported as capture-vs-duty-cycle curves under a
   leakage-free, time-blocked split.
4. **An onboard-feasibility analysis** comparing full, retrieval-free, and geometry-only feature
   tiers, day and night, with per-decision compute and memory cost.

## 1.5 Paper roadmap

Section 2 reviews onboard autonomous targeting and dynamic targeting. Section 3 describes the
dataset. Section 4 presents the label finding. Section 5 defines the gating policies and the
metric. Section 6 analyses onboard feasibility. Section 7 reports results. Section 8 discusses
limitations and the path to a sequential RL policy.

---

### Writing notes (delete before submission)

- Keep the EarthCARE-runs-continuously caveat in §1.1. Reviewers will raise it if you don't.
- Name one concrete power-limited target mission in §1.1. Best candidate: INCUS — three small
  satellites, each with a RainCube-heritage Ka-band scanning radar, launch listed for 2027
  [tanelli2023incus]; RainCube itself flew a radar on a 6U CubeSat [peral2019raincube]. Quote a
  power or duty-cycle figure only after reading it in the mission paper.
- Suggested §1.1 sentence: "Radars are now flying on CubeSats [peral2019raincube] and are planned
  for small-satellite constellations [tanelli2023incus]; on such platforms, power and downlink may
  not allow continuous operation."
