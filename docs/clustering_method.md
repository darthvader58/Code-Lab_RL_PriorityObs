# Why Threshold-Based Cloud Clustering Does Not Fit This Problem

**Applies to:** `geos5data_0.5deg.ipynb` (0.5° resolution) and `geos5data_0.0625deg.ipynb`
(0.0625° resolution). Both notebooks implement the same method — a brightness-temperature
threshold followed by connected-component clustering — at different spatial resolutions. This
document explains what that method does, and why it is the wrong tool for the specific decision
problem this project needs a dataset for.

---

## 1. The decision problem

The dataset trains a reinforcement-learning agent to operate EarthCARE, a satellite that carries
two instruments with very different costs. The Multi-Spectral Imager (MSI) is a passive camera:
it runs continuously, cheaply, and images a wide swath below the satellite. The Cloud Profiling
Radar (CPR) is an active instrument: it points straight down (nadir), draws significant power, and
can only be run some of the time.

The agent's job is to decide, at a fixed interval as the satellite flies along its orbit, **whether
to power on the CPR right now.** It makes that decision using only what MSI can see — cloud cover,
cloud thickness, cloud-top temperature, and similar quantities. It is never shown the precipitation
itself; that would be telling it the answer. Afterward, the decision is scored against what the CPR
would actually have measured directly beneath the satellite at that moment.

This is a **per-decision, per-location classification problem.** Every fixed time step along the
orbit needs exactly one row: the MSI-visible conditions at that moment, and the truth of whether
firing the CPR there would have been worthwhile. The dataset needs to be shaped around that unit —
one row per decision — regardless of whether anything unusual is happening in the atmosphere at
that moment.

---

## 2. What the clustering method does

Both clustering notebooks build their dataset in four steps:

1. **Threshold a temperature field.** A brightness temperature is computed for every grid cell in
   a snapshot of the atmosphere and compared against a fixed cutoff, conventionally 220 K. Cells
   colder than the cutoff are marked as "cloudy."
2. **Group connected cells into clusters.** `scipy.ndimage.label` scans the thresholded grid and
   assigns a single integer ID to every group of touching cold cells, the standard connected-component
   labeling algorithm. Each group becomes one object — intended to represent one storm.
3. **Collapse each cluster into a single row.** `geopandas.dissolve` merges every cell belonging to
   a cluster into one polygon, and aggregates that cluster's properties (area, average and peak
   precipitation, cell count) into one summary row.
4. **Label each cluster by whether the satellite passed over it.** The satellite's ground track for
   that time window is intersected against the cluster polygons; a cluster is marked `observed` if
   the track touches it anywhere.

The output is a table where **each row is one storm object**, carrying its size, its average and
peak precipitation, and whether it happened to be observed. This is a standard, well-established
approach in meteorology — it is essentially how the field has tracked mesoscale convective systems
for decades (e.g., the TOOCAN and PyFLEXTRKR tracking frameworks). It answers the question "where
are the storms, and how big are they?"

That is a different question from the one this project needs answered.

---

## 3. Why it doesn't work for this problem

### 3.1 It answers "where is the storm," not "should I fire the radar right now"

This is the fundamental mismatch, and every other problem below is really a symptom of it.

The RL agent needs a decision every fixed interval along its orbit — whether or not a storm object
happens to exist nearby. A storm-object table only produces a row when the threshold happens to
fire and produce a connected region. There is no natural way to turn "here are this hour's storm
objects" into "here is the CPR-on/off decision at minute 47 of the orbit." The unit of the table
(a storm) and the unit the agent needs (a decision) are not the same thing, and no amount of
post-processing reconciles them cleanly — the join between storm objects and satellite passes has
to approximate a relationship (does this pass touch this storm?) that the agent's real decision
does not care about (a storm doesn't need to exist for the agent to make a firing decision; a
precipitating column that never forms a connected component under the threshold still needs to be
correctly labeled "worth firing on").

Everything the clustering method is good at — object size, shape, lifecycle, identity across time —
is information the CPR-gating decision does not need, and everything the decision needs — a clean
row at a fixed cadence with an unambiguous label — is not what a threshold-and-cluster pipeline
naturally produces.

### 3.2 The threshold itself is measuring the wrong physical quantity

Independent of the object-vs-decision mismatch, the 220 K threshold used in the original notebook
has its own problem. The field it is applied to, `lwtup`, is the **broadband** top-of-atmosphere
outgoing longwave radiation — essentially all of the heat the whole atmospheric column is radiating
to space, from the surface up. Converting it to a temperature via the Stefan-Boltzmann law,

$$T_b = \left(\frac{\text{lwtup}}{\sigma}\right)^{1/4}, \qquad \sigma = 5.67\times10^{-8}\ \text{W m}^{-2}\text{K}^{-4},$$

gives a *column effective emission temperature*, not a cloud-top temperature. Those two quantities
only agree when a thick cloud fills the view and radiates like a blackbody at its own top — the
tropical deep-convection case the 220 K threshold was originally calibrated for. Over a clear, cold
surface (an ice sheet, a snow-covered winter continent), the *surface itself* can radiate at a
similarly low effective temperature, with no cloud involved at all:

| scene | outgoing radiation | implied temperature |
|---|---|---|
| the 220 K threshold | 132.8 W/m² | 220.0 K |
| tropical deep convection | ~90–120 W/m² | 199.6 – 214.5 K |
| clear sky over a cold ice sheet | ~100–130 W/m² | 204.9 – 218.8 K |

A deep thunderstorm and a patch of clear Antarctic sky can radiate at the same effective
temperature and land on the same side of the threshold. This is a known failure mode in satellite
meteorology — infrared window channels (a genuinely different, near-transparent part of the
spectrum) are used specifically because broadband flux does not distinguish cold cloud from cold
surface, and even window-channel methods still need extra tests over snow and ice for the same
reason.

This isn't a hypothetical concern. Running the detector against real G5NR data, **30.5% of the
detected clusters fall within 5.0% of Earth's surface — the polar band poleward of 60°S** — a
6.1× enrichment relative to that band's actual area share. That is the pipeline's own output,
not a simulation: it is finding cold ground, not just cold cloud.

### 3.3 Merging cold regions into single objects amplifies the false positives

Connected-component labeling has a structural property that makes the temperature problem worse,
not better. A threshold applied to a continuous field is a *merging* operator: any two cold regions
connected by even a single cold pixel become one object. Over the ocean or scattered convection
this rarely matters. Over a uniformly cold ice sheet, it means the entire cold region — potentially
hundreds of thousands of square kilometers — collapses into a **single row**, carrying that whole
area's average precipitation, which for clear ice is close to zero.

The largest single object visible in the original notebook's own output covers roughly
9.9×10⁵ km² — one to two orders of magnitude larger than a genuinely large storm system. Because a
polar-orbiting satellite crosses every latitude band on every orbit, an object this large is almost
guaranteed to be touched by some pass, so it enters the "observed" class as one large, essentially
zero-precipitation example — while the pass that happened to graze one corner of it gets credited
with the *entire* object's area and total precipitation, not just the small piece it actually flew
over. A pass sweeping at most ~6.5×10⁵ km² can end up credited with 39× more "observed" area than
it could have possibly imaged. This inflates exactly the rows that should be teaching the agent
"there was nothing worth firing on here."

### 3.4 Trying to fix the threshold: a real improvement, but not a solution

It's natural to ask whether simply pointing the threshold at a better variable would resolve this.
That was tried directly: `cldtmp`, the model's own diagnosed cloud-top temperature, is exactly the
kind of quantity the 220 K convention was built for — measured only where a cloud top exists, not
integrated across the whole atmospheric column. Substituting it for the broadband field and running
it against real data gave a genuine improvement:

| | old threshold (`lwtup`) | `cldtmp` threshold |
|---|---|---|
| high-latitude cells flagged | 53.9% | 7.6% (an 86% reduction) |
| tropical cells flagged | 1.1% | 20.6% |

Polar over-detection fell sharply, and tropical detection — previously suppressed to almost nothing
— recovered to a plausible rate. But two things limited how far this could go. First, `cldtmp` turned
out not to be a clean cloud mask on its own: it still carried a valid-looking value over 18% of
cells with essentially no cloud fraction at all, so a naive fill-value check wasn't enough. Second,
and more fundamentally, even a correctly-gated `cldtmp` threshold detects **cold cloud**, not
**convection**. High latitudes genuinely have a great deal of cold cloud — frontal systems, cirrus
— that produces steady stratiform snow rather than the convective precipitation the project actually
cares about. Improving the threshold's target variable fixed the surface-vs-cloud confusion; it did
not, and could not, fix the fact that "cold" and "convective" are not the same thing.

This matters for the argument as a whole: even the best-tuned version of a threshold-and-cluster
detector still answers a *detection* question ("is there cold cloud here?"), and the project's
actual need was never a detection question in the first place (§3.1).

---

## 4. What this means for the dataset

None of this is a criticism of connected-component clustering as a technique — it is a standard,
useful method for the question it is built to answer: *where are the storms, and how big are
they?* That question matters for tracking storm systems over their lifecycle, studying their
size distribution, or building a climatology of convective activity.

It is a different question from *should the CPR be powered on right now, at this exact point along
the orbit?* — a question that needs an answer at a fixed cadence, independent of whether a
connected cold region happens to exist, and that is corrupted rather than helped by summarizing
large, mixed-precipitation areas into single rows. The right fix was not a better threshold or a
better clustering algorithm — it was restructuring the dataset around the actual unit of the
decision. That redesign is documented in [`trackwise_method.md`](./trackwise_method.md).
