# 8. Limitations and Future Work

## 8.1 Limitations (state each one in the paper)

1. **Model diagnostics, not retrievals.** Features are G5NR diagnostics used as idealized proxies for
   imager products. No instrument forward model or retrieval chain was applied. `tautot` is a column
   total; a real MSI optical-depth retrieval describes the top cloud layer and needs daylight.
2. **Resolution.** G5NR is ~7 km and 30 min; MSI pixels are ~500 m. Sub-grid convection and the
   radar's ~1 km footprint are not resolved. Nadir labels are the maximum over model cells, not a
   radar simulation.
3. **One simulated period.** June 2005 – January 2006 of a single nature run. Results may not
   transfer to other years, to reanalysis, or to real observations. G5NR has a documented split
   inter-tropical convergence zone and a weaker-than-observed Hadley circulation
   [gelaro2015g5nr], which may affect where tropical precipitation, and therefore tropical gating
   skill, appears.
4. **The budget is hypothetical for EarthCARE.** EarthCARE's CPR runs continuously. The duty-cycle
   budget models a future power- or data-limited radar, using EarthCARE geometry as a testbed.
5. **Independent decisions.** Each decision is scored on its own. There is no battery state, no
   switching cost, no warm-up time, and no use of radar-only features revealed after firing. These
   are exactly what make the problem sequential, and they are deferred to the RL paper.
6. **Proxy forward view.** `lwtup_fore` is the next decision's swath mean, not a BBR footprint.
7. **Gaps.** 1,161 missing decision IDs (49 discontinuities) are excluded, not filled.
8. **Diurnal coverage.** EarthCARE's sun-synchronous orbit samples each location at nearly fixed
   local times: ~14:00 (descending) and ~02:00 (ascending); 13:56–13:58 and 01:56–01:58 at the
   nodes over the study window. Every rate reported here (positive rate, recall, AUC) is therefore
   conditional on precipitation present at those times, not on daily precipitation. Precipitation
   between the windows, such as late-afternoon convection over land, never enters the data. The
   labels are still correct for the gating task, because the radar can only measure what is beneath
   it when it passes over. But a policy trained only on ~14:00/~02:00 weather may not transfer to a
   radar in an orbit with different local times, including a non-sun-synchronous orbit that samples
   all of them. TRMM and GPM, which pair a radar with passive imagers on non-sun-synchronous orbits,
   are discussed in §8.3.

## 8.2 Future work

1. **Validation on real data.** EarthCARE MSI and CPR products are public. Repeat the core
   experiment on real collocated observations: from MSI (thermal-infrared channels and, by day,
   cloud products), predict CPR-detected precipitation at nadir. An A-Train analogue (MODIS + CloudSat)
   would provide a multi-year second dataset. This is the single most important follow-up.
2. **Sequential RL.** Add a battery/energy state, a switching cost, and the `cpr_conditional`
   features revealed only after firing. The RL policy must beat the best baseline here at the same
   duty cycle; otherwise a thresholded classifier is sufficient.
3. **Finer cadence.** Precipitation decorrelates within one minute at nadir. A 10-second cadence
   would test whether faster gating helps.
4. **A named target mission.** Re-run the budget analysis with the actual power and duty-cycle
   constraints of a specific small-satellite radar.
5. **Quantify the diurnal blind spot.** G5NR has weather every 30 min, so along the same ground
   track, compare precipitation at the sampled times with precipitation at all hours. This gives the
   fraction of daily precipitation the ~14:00/~02:00 windows miss, by region and season. Real
   EarthCARE data cannot measure this. Re-running the baselines with other local-time pairings would
   then test whether gating skill depends on the orbit's local time.
6. **Forward-looking context.** Use imager pixels ahead of the nadir point along track (the MSI
   swath sees forward-of-nadir cells only marginally; a dedicated lookahead would connect this work
   to dynamic targeting).

## 8.3 Other missions that fit the same question, without the fixed local-time limitation

The research question (can passive measurements tell an active radar when to acquire?) is not
specific to EarthCARE. Two NASA/JAXA precipitation missions carry an active radar and passive
imagers on the same spacecraft and fly non-sun-synchronous orbits, so their overpasses drift through
the local day instead of staying near ~14:00 and ~02:00 (Limitation 8). This paper does not build
datasets for them. They are named here as the natural next test cases.

**Accumulated sampling is not continuous sampling.** For both missions, the broader local-time
coverage builds up over many passes and days. A given storm can still form and dissipate between
overpasses, and coverage can be uneven across locations and hours. These orbits reduce the
fixed-local-time restriction; they do not remove temporal sampling limits.

### TRMM (1997–2015)

- **Instruments** [nasa_trmm, kummerow1998trmm]: Precipitation Radar (PR, active, 3-D rainfall);
  TRMM Microwave Imager (TMI, passive, atmospheric water, cloud liquid water, cloud ice,
  precipitation); Visible and Infrared Scanner (VIRS, passive, cloud cover, type, and top
  temperature).
- **Orbit:** non-sun-synchronous, 35° inclination, about a 46-day precession cycle relative to the
  Sun, so observation times move through the local day.
- **Fit to this work:** VIRS → PR keeps the same structure as MSI → CPR here (cloud context informs a
  radar decision). Adding TMI allows a second experiment comparing infrared context alone with
  infrared plus passive microwave.
- **Differences:** coverage is tropical and subtropical only, not near-polar. The instruments have
  different resolutions and swath widths, so spatial matching and radar-footprint labels would have
  to be rebuilt for TRMM rather than obtained by swapping the orbital elements. The mission has
  ended, so this would be a historical-data or simulation study, with selective radar activation as
  a counterfactual.

### GPM Core Observatory (2014–present)

- **Instruments** [nasa_gpm_core, hou2014gpm]: Dual-frequency Precipitation Radar (DPR, active,
  Ku- and Ka-band, 3-D precipitation, including lighter rain and falling snow); GPM Microwave Imager
  (GMI, passive, 13 microwave channels) [nasa_gmi].
- **Orbit:** non-sun-synchronous, about 65° inclination, sampling different local times and reaching
  higher latitudes than TRMM.
- **Fit to this work:** the same decision structure (passive observation → acquire radar or not), on
  one spacecraft, so no cross-satellite collocation is needed.
- **Differences:** GMI already measures precipitation-related microwave signals, unlike the
  cloud-only MSI-like context here. A GPM formulation should therefore reward what the radar adds
  beyond the passive measurement (vertical structure, or uncertainty left after GMI), not the
  detection of precipitation alone. Combined radar-radiometer retrieval products contain the
  withheld radar measurement and must not be used as passive inputs. GMI and DPR observe the same
  location about 67 s apart [nasa_gmi], so the analysis must confirm that the passive observation
  would be available before the radar decision.

### Requirements common to both

- Only passive measurements available before the decision may enter the policy's input. Radar
  measurements stay withheld until the action is scored.
- Labels must follow each radar's own footprint and scanning geometry, as §4 argues for EarthCARE.
- Any energy, switching, or duty-cycle cost must be justified for that mission. Carrying a radar
  does not mean rapid, policy-controlled switching is operationally supported.

These missions are candidate extensions. This paper does not show that its dataset, policies, or
budget model transfer to TRMM or GPM.

## 8.4 Closing paragraph (draft)

> This benchmark fixes the label, the split, and the metric for budget-constrained radar gating, and
> reports how much a static, imager-driven gate achieves. A sequential reinforcement-learning policy
> is worth building only to the extent that it closes the remaining gap to the oracle at the same
> budget, using inputs a spacecraft can actually compute.
