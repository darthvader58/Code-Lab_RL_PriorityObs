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
   all of them.

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

## 8.3 Closing paragraph (draft)

> This benchmark fixes the label, the split, and the metric for budget-constrained radar gating, and
> reports how much a static, imager-driven gate achieves. A sequential reinforcement-learning policy
> is worth building only to the extent that it closes the remaining gap to the oracle at the same
> budget, using inputs a spacecraft can actually compute.
