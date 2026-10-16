# 2. Related Work

Target length in the paper: 1.5–2 pages. This section doubles as the condensed review. Every
citation key refers to `09_references.md`; verify each one before use.

## 2.1 Ground-planned vs. onboard-planned observation

Traditional Earth-observation scheduling happens on the ground: requests are planned, uplinked,
and executed open-loop. Onboard planning moves the decision to the spacecraft so it can react to
what it sees. The canonical example is the Autonomous Sciencecraft Experiment on EO-1, which used
onboard event detection and the CASPER planner to retask the spacecraft [chien2005eo1].

**Point to make:** the decision in this paper must be onboard. With a one-minute cadence and
precipitation decorrelating within about a minute at nadir (Section 3), a ground loop is far too
slow.

## 2.2 Dynamic targeting with a lookahead sensor

JPL's Dynamic Targeting (DT) uses a lookahead sensor to image the scene ahead of the spacecraft,
analyses it onboard, and retargets a primary instrument within the same overflight
[chien2023dt, breitfeld2024isairas]. Scenarios include cloud avoidance (MODIS) and storm hunting
(GPM precipitation). Learning-based DT policies — imitation and reinforcement learning trained
against a dynamic-programming expert — outperform the best heuristic by 10.0% and 13.7% on average
[breitfeld2025learning]. DT has been demonstrated in flight on the CogniSAT-6 smallsat
[dt2025cognisat6].

**Point to make:** this paper does not claim onboard targeting or RL-for-targeting as new. It
studies a different variant (Section 2.5).

## 2.3 Onboard machine learning hardware

Onboard neural-network inference has flown: Φ-sat-1 ran a cloud-detection CNN on an Intel Movidius
Myriad 2 [giuffrida2022phisat], and CogniSAT-6 ran the DT pipeline onboard [dt2025cognisat6].
Radiation-hardened flight processors (LEON-class) are much slower but still execute small models
easily at low decision rates.

**Point to make:** the cost of the *policy* is negligible; the cost of the *features* (onboard cloud
retrievals) is the real constraint. This motivates the feature tiers in Section 6.

## 2.4 Active sensors and the nadir footprint

CloudSat [stephens2002cloudsat] and EarthCARE [illingworth2015earthcare] carry nadir-only
profiling radars whose footprint is a narrow line inside the much wider imager swath. Small radar
platforms such as RainCube [peral2019raincube] and the planned INCUS constellation of three
small satellites with RainCube-heritage radars [tanelli2023incus] make duty-cycled radar operation a
realistic design question. TRMM [kummerow1998trmm] and the GPM Core Observatory [hou2014gpm] pair
a precipitation radar with passive imagers on one spacecraft, in non-sun-synchronous orbits that
sample all local times; they are the natural extensions of this setup (Section 8.3).

**Point to make:** studies that score targeting policies against gridded or imager-scale truth
implicitly assume the instrument sees the whole scene. For a nadir radar it does not (Section 4).

## 2.5 Simulation testbeds

Observing-system simulation experiments (OSSEs) using nature runs are the standard way to evaluate
observing strategies before a mission exists. The GEOS-5 Nature Run (G5NR) provides 7 km, 30-minute
global fields for two years [gelaro2015g5nr]; it reproduces extremes and the diurnal cycle of land
precipitation well, but has a documented split-ITCZ bias. Mission-architecture tools such as TAT-C
[lemoigne2017tatc] model instrument geometry on such fields.

## 2.6 Positioning table (put this table in the paper)

| | DT (JPL) | This work |
|---|---|---|
| Decision | where to point / what to image | whether an active radar fires (on/off) |
| Constraint | slew time, pointing | power / duty-cycle budget *d* |
| Information source | separate lookahead sensor, 60–90 s lead | co-located imager, ~simultaneous with nadir |
| Truth | scene-scale value (cloud-free area, storm) | precipitation in the radar's nadir footprint |
| Hidden information | none | radar-only features revealed after firing (future work) |
| Platform | CogniSAT-6 (flown) | EarthCARE geometry + G5NR (simulated) |
| Learner | DP expert → IL / RL | static baselines (this paper) → RL (next paper) |

## 2.7 Gap statement (closing sentence of the section)

> To our knowledge, no published benchmark evaluates budget-constrained on/off gating of a nadir
> radar from co-located imager context, scored against the radar's own footprint, nor reports how
> much of that skill survives with features cheap enough to compute onboard.

Re-run the literature search right before submission to make sure this statement still holds.
