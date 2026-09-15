# Enriched track-wise feature set

This document describes the dataset-preparation output
`trackwise_decisions.parquet`. It does not build or train
the reinforcement-learning agent. Each saved row represents one one-minute CPR
decision opportunity. The source has 351,639 rows, 1,161 missing decision IDs,
and 49 non-consecutive decision discontinuities. All task artifacts are aligned
by the `decision` key; no row-position alignment is used.

## Scientific framing

The cloud and radiation columns are GEOS-5 Nature Run model diagnostics named
after physical cloud variables. They are not simulated EarthCARE retrievals.
No instrument forward model or retrieval chain has been applied. They should be
treated as idealized proxies for the information available to a future policy,
not as measurement-equivalent products.

## Feature tiers

The complete machine-readable registry is [`column_registry.yaml`](./column_registry.yaml).

| Tier | Meaning | Columns |
|---|---|---|
| `always` | Candidate context available at every decision | `cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`, `lwtup_fore`, `solar_hour`, `lat`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` |
| `cpr_conditional` | Stored diagnostics available to a future agent only after a previous CPR action reveals them | `tauhgh_mean`, `taulow_mean`, `cldhgh_mean`, `cldmid_mean`, `cldlow_mean` |
| `target` | Radar/truth quantities used for scoring or analysis, never ordinary agent inputs | all `prectot`, `preccon`, `iwp`, and `lwp` swath/nadir columns |
| `metadata` | IDs, timestamps, raw geometry, cell counts, and validity flags | `decision`, `field`, `g5_step`, `time`, raw `lon`, `n_swath_cells`, `n_nadir_cells`, and the eight `*_valid` flags |

The conditional diagnostics are not called “non-mission-retrievable.” They are
simply not available from MSI and are excluded from the always-available
observation set. They remain idealized conditional information for the future
RL simulation. ATLID could in principle provide some of them, but ATLID draws
310 W versus CPR's 308 W, so this project does not treat ATLID as an always-on
context sensor.

`tautot_mean` and `tautot_max` are marked daylight-only because realistic MSI
optical-depth retrievals require daylight. In addition, MSI optical thickness
would describe the topmost observed cloud, while model TAUTOT is a column total;
the two diverge in multilayer scenes.

## `lwtup_fore`: forward-region proxy

The corrected construction is:

```text
valid_fore = (
    decision.shift(-1) - decision == 1
    and g5_step.shift(-1) == g5_step
    and lwtup_mean.shift(-1) is finite
)
lwtup_fore = lwtup_mean.shift(-1) where valid_fore
```

It is a geometric forward-region proxy for BBR: the next decision samples a
region that the fore view may observe ahead of the spacecraft. It is not a true
BBR retrieval or footprint. It uses the next decision's MSI-swath-style GEOS-5
mean, restricted to the same G5NR weather step, whereas a real BBR fore view has
different footprint geometry and approximately 10 km resolution. The same-step
guard prevents the feature from silently crossing a weather-model update.

For the corrected output, 339,859 values are finite and 11,780 are NaN:
1 final-row NaN, 49 gap-boundary NaNs, 11,717 G5NR weather-step-boundary NaNs,
and 13 NaNs caused by a missing next `lwtup_mean`. `field` and `g5_step`
transitions agree for all observed transitions.

## Missing-target hygiene

The eight validity flags distinguish missing or corrupt targets from legitimate
zeros. The audited source counts are:

| Target | Existing NaNs | Finite corrupt values | Legitimate zeros |
|---|---:|---:|---:|
| `prectot_swath` | 26 | 0 | 5,323 |
| `prectot_nadir` | 24 | 0 | 9,204 |
| `preccon_swath` | 10 | 0 | 183,398 |
| `preccon_nadir` | 10 | 0 | 209,331 |
| `iwp_swath` | 0 | 9 | 25,816 |
| `iwp_nadir` | 0 | 6 | 37,026 |
| `lwp_swath` | 0 | 6 | 13,257 |
| `lwp_nadir` | 0 | 3 | 19,694 |

The 24 finite corrupt values are identified by the extraction hygiene rule
`finite and abs(value) >= 1e6`, which catches float32 fill/sentinel values. They
are converted to NaN in the enriched parquet, and their validity flags are
false. There are 94 invalid entries in total (70 pre-existing NaNs plus 24
finite corrupt values). The validity flags mark 51 unique decisions with at
least one invalid target; the 24 finite corrupt values themselves occur in 15
unique decisions. No imputation occurred.
This matters because `target > threshold` evaluates NaN as `False` in ordinary
boolean code, silently turning missing truth into an apparent negative unless
the validity flag is checked first.

## Land and water fractions

The static source is the one-time cached G5NR
`const_2d_asm_Nx` collection. No time-varying weather field was downloaded and
no weather re-extraction was performed. The fetched axes are exactly 2,881
latitudes from -90 to 90 and 5,760 longitudes from -180 to 179.9375 degrees.

The four source fractions form an exclusive partition to float32 precision:
minimum 0.9992626, maximum 1.0007544, mean 0.9996681, and no cells exceeded an
absolute partition error of 0.01. The data show that `frland` excludes land
ice: over Antarctica, `frland + frocean + frlake` averages 0.3811, while adding
`frlandice` gives 0.9996; Greenland gives 0.5820 versus 0.9997. Therefore:

```text
land = frland + frlandice
water = frocean + frlake
```

The feature calculation uses the cached true nadir orbit propagation, with all
60 one-second samples per decision. It aligns by the actual decision ID, not by
row position, so the missing decision IDs cannot shift later orbit samples.
Both output fractions are continuous values in [0, 1]. The land-fraction mean
is 0.3388 and the water-fraction mean is 0.6608; the mean sum is 0.9997. The
distribution is strongly bimodal: 207,132 decisions are in the [0, 0.1) bin
and 94,506 in [0.9, 1.0), with 17.09% mixed-surface decisions, defined as
`0.05 < land_frac_nadir < 0.95`. Because each decision averages 60 one-second
nadir samples over a long along-track segment, these are not necessarily
individual shoreline grid cells. The orbit-sampled mean land fraction is
0.338833. This is plausible for a near-polar ground track, but it is not
directly comparable to an area-weighted global land fraction because the orbit
does not sample Earth uniformly by surface area.

## Cyclic longitude

The raw longitude is retained as metadata. The always-available cyclic features
are:

```text
sin_lon = sin(lon * pi / 180)
cos_lon = cos(lon * pi / 180)
```

They remove the artificial distance between -179 and +179 degrees. Both values
remain within [-1, 1], and the maximum error in `sin_lon² + cos_lon² = 1` is
2.22e-16. Geographic position features carry memorization risk, so a future
evaluation should include held-out longitude sectors and a geometry-only
ablation.

## Autocorrelation diagnostic

Only finite pairs satisfying `decision[i] - decision[i-k] == k` were used.
This excludes all pairs crossing the 49 decision discontinuities.

| Variable | k=1 (pairs) | k=2 (pairs) | k=3 (pairs) | k=5 (pairs) | k=10 (pairs) | k=30 (pairs) |
|---|---:|---:|---:|---:|---:|---:|
| `preccon_nadir` | 0.1160 (351,575) | 0.0566 (351,522) | 0.0385 (351,469) | 0.0249 (351,395) | -0.0032 (351,224) | -0.0161 (350,653) |
| `prectot_nadir` | 0.1712 (351,555) | 0.0603 (351,499) | 0.0160 (351,441) | 0.0112 (351,369) | 0.0059 (351,199) | -0.0078 (350,628) |
| `tautot_mean` | 0.6285 (351,557) | 0.3701 (351,501) | 0.2300 (351,445) | 0.0668 (351,367) | -0.0544 (351,194) | -0.0307 (350,623) |

The precipitation series fall below 1/e by the first sampled lag, showing weak
precipitation persistence at one-minute spacing. `tautot_mean` crosses the
1/e level between approximately lags 2 and 3 and retains structure for roughly
two to three steps. This does not prove every sequential feature is useless; it
does show that carrying a precipitation observation forward at this cadence is
likely stale. A 10-second cadence is only a possible future experiment and was
not tested; it would require re-extraction.

## For the RL phase — not implemented in dataset preparation

The following are design requirements for a future environment, not behavior
baked into this parquet:

- CPR information revealed by action `t` may affect observations at `t+1` or
  later, never the decision or score at `t`.
- Every conditional feature needs its value, age, and availability mask. Age
  should be bounded through a decay such as `exp(-age/tau)` or
  `1/(1+age)`; raw unbounded age should not be fed to the agent.
- Availability must be generated during rollout from the agent's own CPR
  actions. Never bake a label-dependent availability mask into the parquet.
- Targets are never observation features.
- A fire-never versus fire-always rollout is a useful bounding experiment before
  implementing conditional-memory channels.

## Limitations

The dataset retains 1,161 missing decision IDs and 49 discontinuities. GEOS-5
diagnostics are proxies rather than EarthCARE retrievals; no forward model was
applied. `lwtup_fore` approximates BBR forward geometry but is not a BBR
footprint. Precipitation persistence is weak at one-minute spacing. Realistic
MSI optical-depth retrievals are daylight-limited. Same-minute decision-time
causality is not resolved by dataset preparation and must be enforced by the RL
environment. Orbit alignment uses the fixed TLE and cached one-second
propagation, so future orbit-estimation changes would require a new geometry
calculation.
