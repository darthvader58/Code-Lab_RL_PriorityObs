# 8. Limitations and Future Work

## 8.1 Limitations (state each one in the paper)

1. **Model diagnostics, not retrievals.** Features are G5NR diagnostics used as idealized proxies for
   passive-sensor products. No instrument forward model or retrieval chain was applied. `tautot` is a
   column total; a real MSI or VIRS optical-depth retrieval describes the top cloud layer and needs
   daylight. G5NR produces no microwave brightness temperatures at all, which is why every mission's
   passive context is visible/infrared and the microwave imagers (TMI, MWRI-RM) are not used (§3.8).
2. **Resolution.** G5NR is ~7 km and 30 min; MSI pixels are ~500 m, and the radar footprints range
   from ~1 km (CPR) to ~5 km (PR, PMR). Sub-grid convection is not resolved. Labels are the maximum
   over model cells, not a radar simulation.
3. **One simulated period.** June 2005 – January 2006 of a single nature run, for all three
   missions. Results may not transfer to other years, to reanalysis, or to real observations. G5NR has
   a documented split inter-tropical convergence zone and a weaker-than-observed Hadley circulation
   [gelaro2015g5nr], which may affect where tropical precipitation, and therefore tropical gating
   skill, appears. This matters most for TRMM, whose coverage is entirely tropical and subtropical.
4. **The budget is hypothetical.** The CPR, PR, and PMR all run(ran) continuously. The duty-cycle
   budget models a future power- or data-limited radar, using these missions' geometry as testbeds.
5. **Independent decisions.** Each decision is scored on its own. There is no battery state, no
   switching cost, no warm-up time, and no use of radar-only features revealed after firing. These
   are exactly what make the problem sequential, and they are deferred to the RL papers.
6. **Proxy forward view.** `lwtup_fore` is the next decision's swath mean, not a BBR footprint.
7. **Three cases, not a general result.** Each mission is trained and validated separately, and the
   three can give different answers. Agreement across them supports a general conclusion; it does
   not prove one. Results for any other radar mission still need that mission's own dataset.
8. **Diurnal coverage.** EarthCARE's sun-synchronous orbit samples each location at nearly fixed
   local times: ~14:00 (descending) and ~02:00 (ascending); 13:56–13:57 and 01:56–01:57 at the
   nodes over the study window, with 82% of minutes in the six local hours around them. Every EarthCARE rate (positive rate, recall, AUC) is therefore
   conditional on precipitation present at those times, not on daily precipitation. Precipitation
   between the windows, such as late-afternoon convection over land, never enters the EarthCARE data.
   The labels are still correct for the gating task, because the radar can only measure what is
   beneath it when it passes over. TRMM and FY-3G, in precessing orbits, sample all local times, so
   the paper measures this limitation directly (E12) rather than only stating it. Their coverage of
   the day is accumulated over many days; a given storm can still form and dissipate between passes.
9. **Latitude coverage.** TRMM's 35° orbit covers only the tropics and subtropics, and FY-3G's 50°
   orbit reaches the lower mid-latitudes. Poleward of ~50°, only EarthCARE samples, at its two local
   times, so the diurnal question (Limitation 8) stays open for high-latitude precipitation.
   Cross-mission comparisons must be restricted to the latitudes the compared missions share or
   reported by latitude band.
10. **Visible/infrared context only.** All three gates see clouds, not rain. Missions whose passive
    sensor already sees precipitation (a microwave imager) pose a different
    question, how much the radar adds beyond it, which this paper does not answer. FY-3G's 2025–26
    orbit is also paired with 2005–06 weather by day of year, as EarthCARE's is, and its two orbit
    raises in the window move the track along-orbit once each (`../FY3G_trackwise_method.md` §4).
11. **Policy timing was measured on a laptop, not on flight-like hardware (assumption).** The
    per-decision cost in §6.2 comes from Python on a laptop CPU (about 1 ms per decision, mostly
    interpreter overhead) plus counted operations (at most ~5,000 multiply-adds or ~3,350 tree
    comparisons). The claim that the policies fit onboard rests on two assumptions: that a flight
    processor is no more than a few hundred times slower than the laptop, and that the policy would
    be compiled (C or fixed-point) rather than run in Python. A radiation-hardened LEON-class
    processor (tens to hundreds of MIPS) satisfies the first with a wide margin against the 60 s
    decision interval, but this was not measured. Memory (3 kB to 1 MB serialized) was not checked
    against a specific flight computer either. For a real-life use case, the policies should be timed
    on representative hardware before any claim of onboard readiness (future work 7).

## 8.2 Future work

1. **Validation on real data.** EarthCARE MSI and CPR products are public. Repeat the core
   experiment on real collocated observations: from MSI (thermal-infrared channels and, by day,
   cloud products), predict CPR-detected precipitation at nadir. An A-Train analogue (MODIS + CloudSat)
   would provide a multi-year second dataset. FY-3G's PMR and MERSI-RM are also collocated on one
   spacecraft; if their products are accessible, they would give a real-data scanning-radar case.
2. **Sequential RL, one policy per mission.** Add a battery/energy state, a switching cost, and the
   radar-conditional features revealed only after firing. Each mission's RL policy must beat that
   mission's best baseline here at the same duty cycle; otherwise a thresholded classifier is
   sufficient.
3. **Finer cadence.** Precipitation decorrelates within one minute at nadir. A 10-second cadence
   would test whether faster gating helps.
4. **A named target mission.** Re-run the budget analysis with the actual power and duty-cycle
   constraints of a specific small-satellite radar, e.g., INCUS, whose scanning Ka-band radars are
   closest to the TRMM and FY-3G cases.
5. **Quantify the diurnal blind spot.** G5NR has weather every 30 min, so along EarthCARE's ground
   track, compare precipitation at the sampled times with precipitation at all hours. This gives the
   fraction of daily precipitation the ~14:00/~02:00 windows miss, by region and season. Real
   EarthCARE data cannot measure this.
6. **Forward-looking context.** Use imager pixels ahead of the nadir point along track (the MSI
   swath sees forward-of-nadir cells only marginally; a dedicated lookahead would connect this work
   to dynamic targeting).
7. **Timing on representative hardware.** Export each policy to a compiled form (e.g., C code
   generated from the logistic-regression weights, a LightGBM model compiled with Treelite, a
   fixed-point MLP) and time it on a low-power board: a Raspberry Pi–class single-board computer as
   an accessible first step, then a flight-representative processor or its emulator (e.g., a LEON3/
   LEON4 board, or an FPGA soft core) if the work moves toward a mission. Report time per decision,
   peak memory, and energy per decision, alongside the cost of computing the input features, which
   is expected to dominate (§6.1).
8. **Value prioritization with microwave context.** Gate a radar from a passive sensor that already
   sees rain: TRMM with TMI, or FY-3G with MWRI-RM. The label must then score what
   the radar adds beyond the passive sensor (vertical structure, light rain and snow, or the passive
   retrieval's uncertainty), and the inputs need real data or a microwave forward model on the
   nature run (§3.8).

## 8.3 Mission staging and requirements

The three missions are separate application cases, done one at a time (advisor's recommendation:
depth on one case before breadth). EarthCARE is the primary case and is completed first. TRMM is
second: it reuses almost the whole EarthCARE pipeline (cloud-proxy passive features on G5NR) and its
2005 orbit matches the G5NR year. FY-3G is third: it reuses TRMM's pipeline and label unchanged, with
a wider swath and a 50° orbit. Working one mission at a time also
produces a repeatable method; the later missions may still expose problems that require changes to
it.

Requirements common to all three:

- Only passive measurements available before the decision may enter the policy's input. Radar
  measurements stay withheld until the action is scored.
- Labels must follow each radar's own footprint and scanning geometry (Section 4).
- Any energy, switching, or duty-cycle cost must be justified for that mission. Carrying a radar does
  not mean rapid, policy-controlled switching is operationally supported.
- The orbit must come from that mission's historical TLE set, matched to G5NR by UTC time of day and
  day of year.

If TRMM or FY-3G cannot be completed at the same depth as EarthCARE before submission, the paper is
submitted as EarthCARE-only, and the unfinished missions move to a follow-up paper
(`14_submission_plan.md`).

## 8.4 Closing paragraph (draft)

> Autonomous prioritization of Earth-science observations needs benchmarks before it needs
> learners. These three benchmarks fix the label, split, and metric for prioritizing a spaceborne
> radar's observations on missions with different orbits, radar geometries, and passive sensors, and
> report how much a static, passive-driven policy achieves on each. A sequential
> reinforcement-learning policy is worth building only to the extent that it closes the remaining gap
> to the oracle at the same budget, on that mission, using inputs the spacecraft can actually compute.
