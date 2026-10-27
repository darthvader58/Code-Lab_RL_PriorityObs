# 0. Title, Abstract, Keywords

## Title options

1. *Benchmarks for Autonomous Prioritization of Spaceborne Radar Observations: EarthCARE, TRMM, and
   FY-3G* (preferred if all three missions are completed)
2. *Can a Passive Imager Tell a Nadir Radar When to Fire? A Track-Wise Benchmark from EarthCARE
   Geometry and the GEOS-5 Nature Run* (fallback if the paper is EarthCARE-only)
3. *Passive-Informed Duty Cycling of Spaceborne Precipitation Radars Across Three Mission Geometries*
4. *What a Passive Sensor Can Tell a Radar: Budget-Constrained Gating Benchmarks Across EarthCARE,
   TRMM, and FY-3G*

## Abstract (draft — ~250 words; EarthCARE numbers final, replace every [TO RUN] before submission)

Autonomous spacecraft will need to decide onboard which Earth-science observations to prioritize as
conditions below them change. Spaceborne cloud and precipitation radars are a clear case: they return
information passive sensors cannot, but they are expensive to operate, and on power-limited future
platforms a radar may only run for a fraction of each orbit. We ask what baseline skill static
policies achieve in prioritizing when to activate a spacecraft's radar using only its passive sensors,
and how its orbit and radar footprint shape that baseline. The passive sensors see only clouds, so
the decision is detection prioritization: will the radar find precipitation now? We introduce three
track-wise benchmarks for this decision, one for each of three missions that pair a radar with a
visible/infrared imager on one spacecraft and together sample every local time to ±50° latitude and
two local times beyond it: EarthCARE (nadir cloud radar, sun-synchronous orbit, two local times), TRMM (scanning
precipitation radar, 35° precessing orbit), and FY-3G (dual-frequency scanning radar, 50° precessing
orbit). For each mission, one-minute radar on/off
opportunities are generated along the real ground track from historical orbital elements over eight
months of the 7 km GEOS-5 Nature Run, matched by time of day and season. Each mission is trained and
evaluated separately. We show that scoring EarthCARE's nadir radar against imager-swath truth more
than doubles the positive class (13.7% to 30.0% of minutes at ≥ 1 mm/hr), against 1.5× for TRMM's
scanning radar. Under a leakage-free, time-blocked split, a passive-driven
prioritization policy captures 66% of rainy minutes on EarthCARE (AUC 0.89), 43% on TRMM (AUC 0.82–0.85,
where rain fills 41% of minutes and even an oracle reaches only 49%), and [TO RUN]% on FY-3G at a 20%
duty cycle, against 28% for geometry-only gating on EarthCARE and 20% for random gating. On EarthCARE,
night-capable inputs retain 89% of that skill. Comparing the missions shows how
orbit local-time and latitude sampling and radar geometry shape achievable gating skill; a policy
trained on EarthCARE's two local times loses at most 2.5 points on TRMM [FY-3G TO RUN]. The three benchmarks fix the label, split, metric, and
baseline that a sequential reinforcement-learning policy for each mission must beat, as a step toward
autonomous prioritization of Earth-science observations. Datasets and code are released at [DOI].

## Keywords

onboard autonomy; autonomous observation prioritization; dynamic targeting; spaceborne radar; EarthCARE; TRMM; FY-3G; duty cycling;
precipitation; observing system simulation; benchmark dataset; machine learning

## Plain-language summary (needed for AGU/EGU-style venues; ~110 words)

Satellites will increasingly need to decide for themselves which observations matter most. Weather
radars on satellites see inside storms, but they use a lot of power, and future satellites may only
be able to turn their radar on part of the time. We built datasets for three real satellites
that carry both a radar and passive sensors, asking minute by minute whether turning the radar on
would have caught precipitation, using only what the passive sensors could see. Each satellite is
studied separately, because each sees the Earth at different times of day, with a different radar
footprint, and over a different range of latitudes. We report how much better than chance simple
models do on each, how much survives with inputs a small onboard computer can handle, and how the
three satellites differ.
