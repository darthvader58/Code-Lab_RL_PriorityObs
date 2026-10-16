# 0. Title, Abstract, Keywords

## Title options

1. *Can a Passive Imager Tell a Nadir Radar When to Fire? A Track-Wise Benchmark for
   Budget-Constrained Radar Gating* (preferred)
2. *Imager-Driven Duty Cycling of Spaceborne Precipitation Radars: A Benchmark and Baselines*
3. *Where Should a Power-Limited Radar Look? Nadir-Truth Baselines from EarthCARE Geometry and the
   GEOS-5 Nature Run*

## Abstract (draft — ~230 words; replace every [TO RUN] before submission)

Spaceborne precipitation radars return information passive imagers cannot, but they are expensive
to operate. On power-limited future platforms, such as small-satellite radar constellations, a radar
may only run for a fraction of each orbit, and the choice of when to run it determines how much
precipitation it observes. We introduce a track-wise benchmark for this decision: 351,639 one-minute
radar on/off opportunities along EarthCARE's real ground track over eight months of the 7 km GEOS-5
Nature Run, each pairing imager-visible context with precipitation truth in the radar's nadir
footprint and across the imager swath. We show that scoring against swath-scale truth more than
doubles the positive class (28.9% vs. 13.0% at ≥1 mm h⁻¹), and that 55.2% of swath-positive minutes
contain no qualifying precipitation under the radar. Under a leakage-free, time-blocked split, an
imager-driven gate captures [TO RUN]% of nadir precipitation events while the radar operates
[TO RUN]% of the time, against [TO RUN]% for geometry-only gating and [TO RUN]% for random gating.
A retrieval-free feature set that works day and night retains [TO RUN]% of this skill, and the
policy costs on the order of [TO RUN] operations per decision, well within flight-processor budgets.
These baselines fix the label, split, and metric that a future sequential reinforcement-learning
policy must beat. The dataset and code are released at [DOI].

## Keywords

onboard autonomy; dynamic targeting; spaceborne radar; EarthCARE; duty cycling; precipitation;
observing system simulation; benchmark dataset; machine learning

## Plain-language summary (needed for AGU/EGU-style venues; ~100 words)

Weather radars on satellites see inside storms, but they use a lot of power. Future small
satellites may only be able to turn their radar on part of the time. We built a dataset that asks,
minute by minute along a real satellite's path, whether turning the radar on would have caught rain
directly below it, using only what the satellite's camera could see. We find that simple models
using camera information choose much better moments than chance, that much of the benefit survives
with cheap inputs a small computer can handle, and that a common shortcut in scoring these choices
overstates how much rain the radar would actually see.
