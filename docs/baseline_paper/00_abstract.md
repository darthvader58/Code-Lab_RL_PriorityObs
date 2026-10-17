# 0. Title, Abstract, Keywords

## Title options

1. *What a Passive Sensor Can Tell a Radar: Budget-Constrained Gating Benchmarks Across EarthCARE,
   TRMM, and GPM* (preferred if all three missions are completed)
2. *Can a Passive Imager Tell a Nadir Radar When to Fire? A Track-Wise Benchmark from EarthCARE
   Geometry and the GEOS-5 Nature Run* (fallback if the paper is EarthCARE-only)
3. *Passive-Informed Duty Cycling of Spaceborne Precipitation Radars Across Three Mission Geometries*
4. *Can Passive Sensors Tell a Radar When to Fire? Track-Wise Benchmarks on EarthCARE, TRMM, and GPM*

## Abstract (draft — ~250 words; replace every [TO RUN] before submission)

Spaceborne cloud and precipitation radars return information passive sensors cannot, but they are
expensive to operate. On power-limited future platforms, a radar may only run for a fraction of each
orbit, and the choice of when to run it determines what it observes. We ask how a mission's orbit,
radar footprint, and passive sensing determine how much a passive-informed gate can capture. We
introduce three track-wise benchmarks for this decision, one for each of three missions that pair a radar with passive sensors on one
spacecraft: EarthCARE (nadir cloud radar, visible/infrared imager, sun-synchronous orbit), TRMM
(scanning precipitation radar, visible/infrared scanner, 35° precessing orbit), and GPM (dual-frequency
scanning radar, microwave imager, 65° precessing orbit). For each mission, one-minute radar on/off
opportunities are generated along the real ground track from historical orbital elements over eight
months of the 7 km GEOS-5 Nature Run, matched by time of day and season. Each mission is trained and
evaluated separately. We show that scoring EarthCARE's nadir radar against imager-swath truth roughly
doubles the positive class [TO RERUN], while the effect for scanning radars is [TO RUN]. Under a
leakage-free, time-blocked split, a passive-driven gate captures [TO RUN]% (EarthCARE), [TO RUN]%
(TRMM), and [TO RUN]% (GPM, measured as the radar's added value beyond the microwave imager) at a
[TO RUN]% duty cycle, against [TO RUN]% for geometry-only gating. Comparing the missions shows how
orbit local-time sampling, radar geometry, and the information already in the passive sensor shape
achievable gating skill [TO RUN]. The three benchmarks fix the label, split, metric, and
baseline that a sequential reinforcement-learning policy for each mission must beat. Datasets and code are released at [DOI].

## Keywords

onboard autonomy; dynamic targeting; spaceborne radar; EarthCARE; TRMM; GPM; duty cycling;
precipitation; observing system simulation; benchmark dataset; machine learning

## Plain-language summary (needed for AGU/EGU-style venues; ~110 words)

Weather radars on satellites see inside storms, but they use a lot of power. Future satellites may
only be able to turn their radar on part of the time. We built datasets for three real satellites
that carry both a radar and passive sensors, asking minute by minute whether turning the radar on
would have caught precipitation, using only what the passive sensors could see. Each satellite is
studied separately, because each sees the Earth at different times of day, with a different radar
footprint, and with different passive information. We report how much better than chance simple
models do on each, how much survives with inputs a small onboard computer can handle, and how the
three satellites differ.
