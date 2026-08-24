# A Methodological Critique of the G5NR-Derived Training Dataset for Reinforcement-Learning-Based Earth-Observation Target Prioritization

**Subject of review:** `geos5data_0.5deg.ipynb` (27 cells), with reference to
`geos5data_0.5deg_2yr.ipynb`, `geos5data_0.5deg_tessellation_8mo.ipynb`, and `A2C_Model.ipynb`
in the repository `Code-Lab_RL_PriorityObs`.

**Scope:** dataset construction only. No code was modified in producing this document.

---


> **Note (added later).** This critique targets the original clustering notebook.
> The pipeline has since been redesigned twice more — see `method_rationale.md` for
> the intermediate 0.5° tessellation approach and `trackwise_redesign.md` for the
> current 0.0625°/30-min track-local dataset, which resolves Defects 1-4 discussed
> below by construction (no threshold-based object detection remains) and implements
> the CPR-gating framing that §7's revision converges on.

## Abstract

This document audits the data-generating pipeline that produces the training table for a
reinforcement-learning (RL) agent that decides, pass by pass, whether to power up EarthCARE's
Cloud Profiling Radar on the basis of what the continuously-operating Multi-Spectral Imager
(MSI, 150 km swath) can see (§7.2). The pipeline derives convective targets
from the NASA GEOS-5 Nature Run (G5NR) Observing System Simulation Experiment (OSSE) archive
at 0.5° resolution (collection `tavg01hr_2d_met3_Cx`) and labels them against TAT-C-computed
ground tracks.

Six defects are documented, four of them severe enough to invalidate downstream learning as
currently posed. (1) The target detector thresholds a *broadband* top-of-atmosphere upwelling
longwave flux converted to an effective emission temperature, and applies a fixed 220 K cut.
This is algebraically identical to the linear test `lwtup < 132.83 W m⁻²`, which cold clear-sky
polar columns satisfy as readily as tropical anvils. (2) The mitigation applied — restricting
the fetch to latitudes within ±75° — removes 3.41% of Earth's surface area at a cost of 16.62%
of the model's grid rows, and does not exclude the contaminated region. (3) Connected-component
labelling followed by a `dissolve` collapses each contiguous cold region into a single row whose
area- and precipitation-derived features are then credited *in full* to any satellite pass that
touches any part of it, without intersection weighting; this makes the positive class an
area-biased sample dominated by large, near-zero-precipitation polar objects. (4) A date-remapping
expression collapses 5,880 hourly frames onto **24** distinct atmospheric states — the 24 hours of
a single day, 2006-05-20 — replayed 245 times; the 1,354,115-row cluster table contains exactly
5,527 distinct meteorological objects. (5) The project's decision problem is
gating a power-limited Cloud Profiling Radar on or off per pass, for which the implemented binary
action space and the exclusion of the label from the observation are both correct design; but the
one RL environment that could be read (`A2C_Model.ipynb`) advances its state index unconditionally,
so no action consumes any resource. The gating problem is thereby reduced to a contextual bandit in
which firing the instrument is free and the duty cycle is fixed at ≈100% by the ratio of two
hand-chosen reward constants. Independent measurement over 2,270,592 cell-hours confirms the
decision is nonetheless learnable (ROC-AUC 0.935 at a 1.94% base rate, 8.4× lift), and that the
field the detector thresholds is the weakest of six candidate predictors. (6) The same OpenDAP collection exposes 71 variables,
including `lwtupclr`, `cldtmp`, `preccon`, `precsno`, `cape`, and the `cld*`/`tau*` families,
which together provide discriminators the current single-field detector cannot express.

Every quantitative claim below is either arithmetic reproduced in-line, or a value read from the
notebook's own stored cell outputs. Departures from the initial hypotheses under review are flagged
explicitly in §11; one finding from an earlier draft of §7 is formally retracted in §7.3.

---

## 1. Introduction and problem setting

### 1.1 The decision problem

An Earth-observing satellite with a narrow instantaneous field of regard cannot image everything
it flies over. EarthCARE's MSI has a 150 km swath (Illingworth et al., 2015; Wehr et al., 2023),
and the satellite completes an orbit in

$$T = \frac{1440\ \text{min day}^{-1}}{15.57041891\ \text{rev day}^{-1}} = 92.483\ \text{min},$$

using the mean motion from the two-line element set hard-coded in cell 0. In one hour the
sub-satellite point traverses

$$L_{1\text{h}} = 40{,}075\ \text{km} \times \frac{60}{92.483} = 25{,}999\ \text{km},$$

so the instrument sweeps roughly

$$A_{1\text{h}} \approx 25{,}999 \times 150 = 3.900\times 10^{6}\ \text{km}^2
= \frac{3.900\times 10^{6}}{5.1006\times 10^{8}} = 0.765\%$$

of Earth's surface per hour, and the fraction of that swath worth committing an expensive active
instrument to is smaller still. Deciding when to commit it is a sequential decision problem under a
power budget, and it is the problem the repository's RL notebooks are meant to address; §7.2 states
the project's formulation precisely. This class of problem — agile
Earth-observation scheduling under resource constraints — has an established RL literature
(Eddy & Kochenderfer, 2020; Herrmann & Schaub, 2023a, 2023b; Chun et al., 2023).

### 1.2 Why dataset construction is the load-bearing step

An RL agent optimizes the reward it is given against the transitions it observes. If the reward
signal is derived from mislabelled targets, the agent will faithfully learn the mislabelling; if
the training distribution contains 24 atmospheric states presented 245 times each, no amount of
algorithmic sophistication recovers the variability that was never sampled. The failure modes
documented here are therefore not "tuning issues" — they set a ceiling on what any downstream
algorithm can achieve, and they do so *silently*, because the resulting table is large (1.35M rows)
and superficially well-formed.

The synthetic-truth source is a well-characterized OSSE archive: the 7 km GEOS-5 Nature Run has
been evaluated for its representation of tropical cyclones and cloud fields for exactly this kind of
observing-system study (Reale et al., 2017). The problem is not the source data; it is the
extraction.

---

## 2. Data and methods as currently implemented

### 2.1 Pipeline structure

| Cell | Function |
|---|---|
| 0 | Defines the `EarthCare` satellite (TLE) and a rectangular `PointedInstrument` "MSI", `swath_width_to_field_of_view(394e3, 150e3, 5.760868)` |
| 1 | `startdate = datetime(2025,7,19,15,4, tzinfo=timezone.utc)`; `duration = timedelta(hours = 5880)`; `step = timedelta(seconds = 5)`; `batch_duration = timedelta(minutes=10)`; computes `ground_tracks` via TAT-C `compute_ground_track` |
| 2 | Reports `35280` ground-track rows |
| 4 | `grid_size = 0.5`; `g5nr_frame_duration = timedelta(hours=1)` |
| 5 | Opens `.../0.5000_deg/tavg/tavg01hr_2d_met3_Cx` over OpenDAP |
| 6 | `lookup_tautot` — attaches `tautot` to each ground-track row |
| 7 | `compute_grid_cell_area(lat, lon)` |
| **8** | **Target detector: threshold, connected-component labelling, dissolve** |
| 9 | Solar hour per cluster centroid (Skyfield) |
| 10 | `observed` label: cluster ∩ hourly ground-track union |
| 12–13 | Per-pass join and aggregation into the final table |
| 15–19 | Diagnostic figures |
| 20–24 | Sanity checks (their outputs are the primary evidence used below) |
| 25 | GeoJSON export |

`duration = 5880 h = 245 days`, and the notebook's own cluster table runs from
`2025-07-19 15:04+00:00` to `2026-03-21 14:04+00:00`, consistent with
`startdate + 5879 h` (244 d 23 h). `35280 = 5880 × 6` ten-minute passes.

### 2.2 The detector (cell 8), verbatim

```python
threshold = 220  # brightness temperature threshold in Kelvin
...
sub = (
    dataset[["lwtup", "prectot"]]
    .sel(time=slice(t_min, t_max), lat=slice(-75, 75))
    .load()
)
...
def _frame_ds_time(frame):
    # same time-mapping trick the original cell uses
    return (startdate + frame * g5nr_frame_duration).replace(
        day=20, month=5, year=2006, tzinfo=None
    )
...
        tb = np.sqrt(np.sqrt(lwtup_2d / 5.67037e-8))
        labels, _ = ndi.label(tb < threshold)
...
        return cells[cells.cluster > 0].dissolve(
            by=["time", "cluster"],
            aggfunc={
                "count": "sum",
                "area": "sum",
                "tot_prectot": "sum",
                "avg_prectot": "mean",
                "max_prectot": "max",
            },
        )
```

with the polygon for each grid cell built as

```python
geometry=[
    box(lo, la, lo + grid_size, la + grid_size)
    for lo, la in zip(lon_vals, lat_vals)
],
```

### 2.3 Recorded outputs used as evidence

All values below are read from the notebook's stored outputs, not recomputed.

| Cell | Output |
|---|---|
| 2 | `35280` ground-track rows |
| 8 | `5880 frames -> 24 distinct g5nr time slices`; `total: 321.5s, 1354115 cluster rows` |
| 20 | `all ground-track passes: 35280`; `observed-only passes: 11837`; `observed fraction: 33.6%` |
| 21/23 | observed clusters per frame: `count 5508`, `mean 3.502`, `std 2.257`, `min 1`, `max 14` |
| 22/24 | 19,290 observed clusters; centroid latitude `mean −30.29`, `50% −39.74`, `25% −66.71`, `min −74.75`, `max 74.75` |
| 24 | band shares: `(−75,−60] 30.5%`, `(−60,−30] 25.4%`, `(−30,0] 18.7%`, `(0,30] 20.3%`, `(30,60] 1.7%`, `(60,75] 3.4%`, `|lat|>75 0.0%` |
| 26 | `11837`, `19978`, `1351910` (rows with `avg_prectot > 0` at three pipeline stages) |

### 2.4 Independently verified properties of the source collection

`curl -s ".../tavg01hr_2d_met3_Cx.dds"` returns 573 lines declaring **71** variables, each of shape
`[time = 18288][lat = 361][lon = 720]`. The sibling notebook records the corresponding DAS time
span as 2005-05-15 21:30Z → 2007-06-16 20:30Z.

---

## 3. Defect 1 — A broadband-OLR threshold cannot separate deep convection from cold surfaces

**Severity: Critical.** This defect determines what a "target" is, so every downstream quantity
inherits it.

### 3.1 Mechanism

`lwtup` is the *broadband* top-of-atmosphere upwelling longwave flux — the vertically integrated
emission of the whole column: surface, water vapour, CO₂, and cloud, weighted by their respective
transmissivities. The line

```python
tb = np.sqrt(np.sqrt(lwtup_2d / 5.67037e-8))
```

inverts the Stefan–Boltzmann law, $T_b = (F/\sigma)^{1/4}$, yielding a *column effective emission
temperature*. It is not a cloud-top temperature. The two coincide only when an optically thick
cloud fills the field of view and radiates as a blackbody near its top — the tropical deep-convective
case for which the proxy was designed (Arkin, 1979; Mapes & Houze, 1993; Machado et al., 1998).

Note first that the quartic root is a strictly increasing bijection on $F > 0$, so the test
`tb < 220` carries exactly the same information as a linear threshold on the raw field:

$$F_{\text{crit}} = \sigma T^4 = 5.67037\times10^{-8} \times 220^4
= 5.67037\times10^{-8} \times 2.34256\times 10^{9} = 132.83\ \text{W m}^{-2}.$$

The detector is `lwtup < 132.83 W m⁻²`. The "brightness temperature" framing adds no discriminative
power; it only makes the threshold appear to be a cloud property.

### 3.2 Quantitative evidence

Inverting $T_b = (F/\sigma)^{1/4}$ across the relevant flux range:

| $F$ (W m⁻²) | 90 | 100 | 110 | 120 | 130 | **132.83** | 150 | 200 | 280 |
|---|---|---|---|---|---|---|---|---|---|
| $T_b$ (K) | 199.6 | 204.9 | 209.9 | 214.5 | 218.8 | **220.0** | 226.8 | 243.7 | 265.1 |

A tropical anvil emitting 90–120 W m⁻² maps to 200–215 K and is detected. A clear-sky column over
the Antarctic plateau whose bulk emission temperature equals a plausible winter surface/inversion
temperature is detected *identically*: a blackbody at 213 K (South Pole winter surface conditions
are of this order; King & Turner, 1997; Turner et al., 2009, document record minima near 184 K)
emits

$$5.67037\times10^{-8} \times 213^4 = 5.67037\times10^{-8} \times 2.0583\times10^{9}
= 116.7\ \text{W m}^{-2} \ \Rightarrow\ T_b = 213\ \text{K} < 220\ \text{K}.$$

This is not a marginal overlap. It is a *complete* one over the range that matters: the detector's
decision boundary sits inside the clear-sky polar distribution.

Two independent lines of evidence support this beyond the algebra.

**(a) Published statements of the same failure mode.** Zhang, Randel & Fu (2016), using OLR as a
convection proxy across three reanalyses, note explicitly that low OLR indicates strong convection
in the tropics but "does not necessarily imply deep convection over high latitudes, where the low
values are largely due to the cold surface and atmospheric temperatures." The satellite
cloud-detection literature reaches the same conclusion from the retrieval side: infrared window
tests fail over snow and ice because surface and cloud brightness temperatures converge, which is
why operational cloud masks add near-infrared, visible, and spectral-difference tests specifically
for polar scenes (Ackerman et al., 1998; Liu et al., 2004; Frey et al., 2008), and why ISCCP applies
a separate clear-restoral test over snow/ice (Rossow & Schiffer, 1999). Yamanouchi & Charlock (1997)
further show that over the Antarctic ice sheet OLR falls at roughly 20 W m⁻² per km of elevation
above 2 km — i.e. the plateau's altitude alone drives OLR into the detection range, independent of
cloud. The physical dependence of clear-sky OLR on column temperature and water vapour is
characterized by Dessler et al. (2008).

**(b) The notebook's own diagnostic.** Cell 24 was written to test precisely this. Its recorded
output, over 19,290 observed clusters:

```
  antarctic    (-75 < lat <= -60)      5881 ( 30.5%)
  S mid-lat    (-60 < lat <= -30)      4892 ( 25.4%)
  S tropics    (-30 < lat <= 0)        3616 ( 18.7%)
  N tropics    ( 0 < lat <= 30)        3913 ( 20.3%)
  N mid-lat    (30 < lat <= 60)         334 (  1.7%)
  arctic       (60 < lat <= 75)         654 (  3.4%)
```

The median observed-cluster centroid latitude is **−39.7°** and the lower quartile is **−66.7°**.
The (−75,−60] band occupies

$$\frac{\sin 75^\circ - \sin 60^\circ}{2} = \frac{0.965926 - 0.866025}{2} = 0.04995 = 5.00\%$$

of Earth's surface, yet supplies 30.5% of detections — an enrichment of

$$\frac{30.5}{5.00} = 6.11\times.$$

The tropics (|lat| < 30°) occupy $\sin 30^\circ = 50.0\%$ of the surface and supply
$18.7 + 20.3 = 39.0\%$ — a *depletion* of $39.0/50.0 = 0.78\times$. The Antarctic-to-tropical odds
ratio is $6.11/0.78 = 7.8$. For a detector whose purpose is to find deep convection, this is the
signature of a detector that is finding cold surfaces.

### 3.3 Consequence for the RL model

The reward signal is derived from `prectot` attached to these clusters, and under the CPR-gating
formulation of §7.2 it defines when firing the radar was the correct call. A polar false positive
enters the table as a "target" with essentially zero precipitation. If the agent is rewarded for
intercepting detected clusters, it learns to fire the CPR over the Southern Ocean and the Antarctic
coast — the worst possible use of a power-limited instrument, since those columns are exactly the
ones with no convective signal to profile. If instead it is rewarded for precipitation, it learns
that a large share of "targets" are worthless, which degrades the signal-to-noise of the reward
and, in a value-based method, inflates the variance of the return estimate without adding
information. Either way the contamination propagates directly into the gating policy.

---

## 4. Defect 2 — The latitude clip is an ineffective mitigation with a real cost

**Severity: Moderate (as an independent defect); it is best read as a symptom of Defect 1.**

### 4.1 Mechanism

Cell 8 restricts the fetch with `.sel(..., lat=slice(-75, 75))`, and cell 8 correspondingly
computes `sample_lat = dataset["lat"].sel(lat=slice(-75, 75)).values`. The intent — visible in the
narrative of cell 24 — is to suppress polar false positives. Because surface area on a sphere
scales as $\sin(\text{latitude})$, a latitude band is a poor lever for excluding polar area.

### 4.2 Quantitative evidence

**Area removed.** The cap poleward of $\phi$ has area $2\pi R^2 (1 - \sin\phi)$. With
$R = 6371$ km, $2\pi R^2 = 2.5503\times10^{8}$ km²:

$$A_{\text{cap}}(75^\circ) = 2.5503\times10^{8} \times (1 - 0.965926)
= 2.5503\times10^{8} \times 0.034074 = 8.690\times10^{6}\ \text{km}^2.$$

Two caps: $1.738\times10^{7}$ km². Earth's surface is $4\pi R^2 = 5.1006\times10^{8}$ km². So

$$\text{area removed} = \frac{1.738\times10^{7}}{5.1006\times10^{8}} = 3.41\%,$$

equivalently $\text{retained} = \sin 75^\circ = 96.59\%$.

**Rows removed.** The DDS gives `lat = 361` (−90 to 90 inclusive at 0.5°). Rows in [−75, 75]:
$(75-(-75))/0.5 + 1 = 301$. So

$$\text{rows removed} = \frac{361 - 301}{361} = \frac{60}{361} = 16.62\%.$$

The clip is therefore **4.9× more expensive in data than in the area it excludes**
($16.62/3.41 = 4.87$).

**It does not exclude the contaminated region.** Antarctica's land area is ≈ $1.40\times10^{7}$ km².
The entire cap poleward of 75°S is only $8.690\times10^{6}$ km², and a substantial part of that cap
is ocean (the Ross, Weddell, and Amundsen embayments). Therefore at minimum

$$1.40\times10^{7} - 8.690\times10^{6} = 5.31\times10^{6}\ \text{km}^2 \quad (\geq 37.9\%\ \text{of Antarctica})$$

lies equatorward of 75°S and is retained — including the entire coastal ring at 66–70°S and the
Antarctic Peninsula to 63°S. Greenland spans roughly 60°N–83.6°N, so only its northern tip is
removed. The winter sea-ice zones of both hemispheres, and all winter continental interiors, lie
equatorward of 75°.

**The notebook's own output confirms the clip is a hard wall, not a filter.** Cell 22 records
`min −74.750000` and `max 74.750000` for observed-cluster centroid latitude: detections are pinned
against the clip boundary. (The value −74.75 rather than −75.00 is a consequence of the polygon
convention discussed in §8.1.) Consequently the two "polar" rows of the cell-24 diagnostic report
`0 ( 0.0%)` **by construction** — the diagnostic is structurally incapable of detecting the residual
contamination it was written to find, and the 30.5% Antarctic share is what leaks past the wall.

### 4.3 Consequence

Data volume: a full 0.5° global field is $361 \times 720 \times 4 = 1.040$ MB. Two variables over
5,880 hours would be 12.23 GB; the clipped version is $301 \times 720 \times 4 = 0.867$ MB per
field, or 10.19 GB — a 16.6% saving in transfer for a 3.4% reduction in the contaminated domain.
The clip trades away 60 rows of genuinely useful high-latitude atmosphere (including real polar
frontal systems that *do* precipitate) without solving the problem it was introduced for.

---

## 5. Defect 3 — Connected-component labelling amplifies rather than dilutes false positives

**Severity: High.**

### 5.1 Mechanism

Three compounding steps:

1. `labels, _ = ndi.label(tb < threshold)` labels the binary mask with `scipy.ndimage`'s default
   structuring element (4-connectivity in 2-D), producing one integer per connected region.
2. `.dissolve(by=["time","cluster"], aggfunc={...})` reduces every region to **one row**, summing
   `count`, `area`, `tot_prectot`, averaging `avg_prectot`, and taking the max of `max_prectot`.
3. Cell 10 labels a cluster `observed` if it intersects the union of that hour's ground-track
   footprints.

A single threshold applied to a continuous field is a merging operator: whenever two systems are
connected by even a one-pixel-wide bridge below 220 K, they become one object. This is the
best-documented pathology in convective-object identification. It is why TOOCAN performs
segmentation in the 3-D space–time domain with an iterative multi-threshold scheme rather than a
single cut (Fiolleau & Roca, 2013, 2024), why PyFLEXTRKR separates a cold-cloud identification step
from a merge/split resolution step (Feng et al., 2023), why TempestExtremes is built around
scale-insensitive pointwise feature definitions instead of fixed-threshold blobs (Ullrich &
Zarzycki, 2017), and why object-based verification frameworks make the threshold-and-merge
sensitivity an explicit tunable (Davis et al., 2006). The MCS-tracking method intercomparison of
Feng et al. (2025) quantifies how much object statistics disagree across algorithms that differ
principally in how they handle exactly this step. In the tropics, a single 220 K cut merges the ITCZ
into continent-scale bands; the size distribution of the resulting objects is then an artefact of
the threshold rather than a property of the atmosphere (Mapes & Houze, 1993; Nesbitt et al., 2006).

### 5.2 Quantitative evidence

**Object size.** A 0.5° grid cell at latitude $\phi$ has area
$A = R^2 \Delta\lambda (\sin(\phi + \tfrac{\Delta}{2}) - \sin(\phi - \tfrac{\Delta}{2}))$.
At $\phi = -75^\circ$ with $\Delta = 0.5^\circ = 0.0087266$ rad:

$$A = 40{,}589{,}641 \times 0.0087266 \times (0.96706 - 0.96468) = 845\ \text{km}^2.$$

The notebook's printed cluster table (first five rows of frame 0) shows components with
`count = 1170` and `count = 556` pixels at −75°, i.e.

$$1170 \times 845 = 9.9\times10^{5}\ \text{km}^2, \qquad 556 \times 845 = 4.7\times10^{5}\ \text{km}^2.$$

For scale, the theoretical ceiling for one component spanning the full retained polar band
(−75° to −60°, all longitudes) is $2.55\times10^{7}$ km², and for the whole retained domain
$4.92\times10^{8}$ km². A large tropical MCS cold-cloud shield below 220 K is of order
$10^4$–$10^5$ km² (Machado et al., 1998; Nesbitt et al., 2006). A single polar object in the very
first frame is therefore already 10–100× the size of a large MCS.

**Size-biased selection into the positive class.** A cluster is `observed` if it intersects the
hour's swath union, which covers 0.765% of the globe (§1.1). For a point-like cluster the hit
probability is ≈ 0.765%; for an object spanning a zonal band it is ≈ 1, because a
97.0168°-inclination orbit crosses every latitude band on every revolution. Empirically,

$$\frac{19{,}290\ \text{observed}}{1{,}354{,}115\ \text{clusters}} = 1.42\%,$$

i.e. $1.42/0.765 = 1.86\times$ the point-like rate — direct evidence that selection into the
positive class is area-weighted. The polar enrichment computed in §3.2 (6.11×) is the same effect
localized: the largest objects are the polar ones, so they are the ones that get observed.

**Whole-cluster features are credited to grazing passes.** This is the most consequential and least
obvious part of the defect. In cell 13,

```python
joined_observed = gpd.sjoin(
    observed_clusters, ground_tracks,
    how="right", predicate="intersects", on_attribute="frame",
)
...
_agg = joined_df.groupby(level=0).agg(
    ...
    count       =("count",       "sum"),
    area        =("area",        "sum"),
    tot_prectot =("tot_prectot", "sum"),
    ...
)
```

`count`, `area`, and `tot_prectot` are *whole-cluster* quantities carried over from the dissolve.
There is no intersection-area weighting anywhere in the pipeline. A 10-minute pass sweeps at most

$$\frac{25{,}999}{6} \times 150 = 6.5\times10^{5}\ \text{km}^2,$$

yet a pass that grazes one corner of a $9.9\times10^{5}$ km² Antarctic component is credited with
that component's entire area, pixel count, and summed precipitation. For a band-spanning component
the mismatch reaches $2.55\times10^{7} / 6.5\times10^{5} = 39\times$ more "observed" area than the
instrument can physically image.

**The positive class inherits the near-zero precipitation.** `avg_prectot` for a dissolved polar
component is the mean over ~$10^3$ near-zero cell values, i.e. ≈ 0, while `area` is $\sim 10^6$ km².
Any downstream normalization by area, or any reward of the form (precipitation per unit observed
area), is therefore dominated by these objects.

### 5.3 Consequence for the RL model

The positive class — the set of rows the agent is trained to seek — is an area-biased sample in
which the largest members carry no reward. In a value-based method this appears as a large mass of
high-`area`, zero-`prectot` transitions; in a policy-gradient method it appears as a systematic
bias in the advantage estimate toward high-latitude passes. The objects also have no identity across
timesteps — a given storm receives a different integer label each hour, since `ndi.label` renumbers
in raster-scan order from scratch every frame. Under the CPR-gating formulation of §7.2 this is not
an action-space problem (§7.3), but it does destabilize the per-pass covariates: `count`, `area`,
`tot_prectot`, `avg_prectot`, and `max_prectot` are aggregates over an object set whose membership
and cardinality change for reasons unrelated to the atmosphere, so nominally identical atmospheric
conditions can present the agent with materially different feature vectors.

---

## 6. Defect 4 — Temporal sampling collapses the dataset to 24 atmospheric states

**Severity: Critical.** This is the defect with the most exactly reproducible evidence and the
narrowest fix.

### 6.1 Mechanism

```python
def _frame_ds_time(frame):
    # same time-mapping trick the original cell uses
    return (startdate + frame * g5nr_frame_duration).replace(
        day=20, month=5, year=2006, tzinfo=None
    )
```

`startdate = 2025-07-19 15:04 UTC`. Advancing by `frame` hours changes the hour (and, on rollover,
the day/month/year). `datetime.replace(day=20, month=5, year=2006)` then overwrites day, month, and
year while preserving hour, minute, and second. The hour field cycles with period 24; the minute is
fixed at 04. The image of the map over `frame ∈ [0, 5880)` is therefore exactly

$$\{\,\text{2006-05-20 } HH\text{:04} : HH = 0, \dots, 23 \,\} \quad (|\cdot| = 24).$$

The notebook prints this itself:

```
5880 frames -> 24 distinct g5nr time slices
```

The same expression is used in cell 6's `lookup_tautot`, so the `tautot` feature attached to every
ground-track row inherits the identical collapse.

### 6.2 Quantitative evidence

**Replication multiplicity.**

$$\frac{5880\ \text{frames}}{24\ \text{states}} = 245\ \text{exact repeats of each hourly field}.$$

**Distinct meteorological objects.** The notebook reports 1,354,115 cluster rows. Then

$$\frac{1{,}354{,}115}{245} = 5{,}527 \quad\text{exactly}\qquad (5{,}527 \times 245 = 1{,}354{,}115).$$

The table contains **5,527 distinct cluster objects**, each replicated 245 times. Consistency check:
$5{,}527 / 24 = 230.3$ objects per atmospheric state, and
$1{,}354{,}115 / 5{,}880 = 230.3$ objects per frame. The two agree, confirming that every frame is a
verbatim re-detection of one of 24 fields.

**Fraction of the archive used.** The DDS declares `time = 18288` hourly steps
(2005-05-15 21:30Z → 2007-06-16 20:30Z). The run nominally requests 5,880 of them
($5880/18288 = 32.2\%$) but realizes 24 ($24/18288 = 0.13\%$).

**Season confound.** 2006-05-20 is day-of-year 140: late austral autumn, with Antarctic sea ice
expanding and the plateau near its seasonal temperature minimum, and boreal late spring. A
97.0168°-inclination sun-synchronous orbit samples the northern and southern high-latitude bands
with essentially identical geometry, so orbital sampling cannot explain a hemispheric asymmetry.
Yet cell 24 reports

$$\frac{\text{S mid-lat} + \text{Antarctic}}{\text{N mid-lat} + \text{Arctic}}
= \frac{25.4 + 30.5}{1.7 + 3.4} = \frac{55.9}{5.1} = 11.0\times.$$

An 11-fold hemispheric asymmetry in cold-cloud "detections" that has no geometric explanation is a
direct measurement of the seasonal confound: the false-positive rate of Defect 1 is being sampled at
the one season that maximizes it in the south and minimizes it in the north, and that season is then
frozen into all 5,880 hours.

**Decoupling of covariates from meteorology.** `solar_hour` (cell 9) is computed from the *real*
2025–2026 timestamp, while `prectot` comes from 2006-05-20. A row at solar hour 14 in December 2025
therefore carries 20 May precipitation. The diurnal-cycle figures in cells 15–16, titled
"Mean Precipitation vs Solar Hour for 1 month period", and the KDE figures in cells 17–19, titled
"for 8 months", all display a single day.

### 6.3 Why this cannot be repaired downstream

**Resampling and augmentation cannot create variability that was never sampled.** The 245 copies of
a given object are byte-identical in every meteorological column. Bootstrapping, SMOTE-style
interpolation, or reweighting operate on the empirical distribution, which has support on 5,527
points; they change the weights on those points, not the support. For the rare heavy-precipitation
class this is decisive: the number of *distinct* heavy-precipitation events in the table is bounded
above by the number of such events present in 24 hours of one day, however many rows the table has.

**The rows are not literal duplicates, and this matters — but only for the geometry.** The
`observed` label *does* differ across the 245 replicas, because the ground-track geometry advances
in real time. That label therefore carries genuine orbital-access information, and any analysis of
*access* (which targets are reachable, how often, at what solar hour) remains valid. What is
degenerate is the *meteorological* conditioning: the joint distribution
$p(\text{precipitation} \mid \text{location}, \text{hour})$ has 24 realizations.

**No temporal holdout can produce a clean evaluation.** The replication cycle has period 24 hours.
Any contiguous train/test split that leaves at least 24 hours on each side places all 24 atmospheric
states in both partitions. `TimeSeriesSplit`, walk-forward validation, and blocked cross-validation
all fail here. This is textbook pseudoreplication (Hurlbert, 1984) and the resulting optimistic
performance estimate is a canonical form of leakage (Roberts et al., 2017; Kapoor & Narayanan, 2023).

### 6.4 This defect is isolated to this notebook

`geos5data_0.5deg_2yr.ipynb` already implements the correct mapping. Its cell 5 reads the real axis:

```python
g5nr_times = pd.to_datetime(dataset["time"].values)
g5nr_start = g5nr_times[0].to_pydatetime()
n_g5nr = len(g5nr_times)
duration = n_g5nr * g5nr_frame_duration
```

and its cell 8 defines

```python
def _frame_ds_time(frame):
    # walk the real g5nr hourly axis; clamp any overshoot to the final slice
    return g5nr_start + min(int(frame), n_g5nr - 1) * g5nr_frame_duration
```

It also fetches over `lat=slice(-89, 89)` rather than `(-75, 75)`, i.e. it does not apply the clip of
Defect 2. The defect is therefore a known one within the project, present in the 0.5° notebook only.

The cost of the fix is real and explains why the shortcut was taken: the fetch scales by 245× (from
24 slabs to 5,880), while the CPU work is unchanged. The 8-month run completed its single chunk in
320.4 s. The sibling notebook's response — 24-hour chunks, per-chunk retries with handle reopening,
and on-disk checkpointing — is the appropriate engineering answer, and a strided fetch (§9, R6)
reduces the volume by a further factor of ~25.

---

## 7. Defect 5 — RL formulation: the decision problem, one retracted finding, and what survives

**Severity: High**, but narrower and differently located than an earlier version of this section
claimed. §7.3 records the retraction explicitly.

### 7.1 What could and could not be read

`A2C_Model.ipynb` was read in full. `DQN.ipynb`, `C51_orbital_timestep.ipynb`,
`Distributional_RL.ipynb`, `Non-Orbital_DQN.ipynb`, `TATC-RL_CoDeLab.ipynb`, and
`RandomForestClassifier_FURI.ipynb` are macOS *dataless* (cloud-evicted) files in this working
copy: `ls -lO` reports the `compressed,dataless` flags, `dd` returns 0 bytes without error,
`brctl download` does not materialize them, and the git object store is likewise unreadable
(`fatal: mmap failed: Operation timed out`). **The analysis below is verified for
`A2C_Model.ipynb` alone.** Claims about the other four RL notebooks are not established.

### 7.2 The decision problem as actually posed: CPR gating

The project's decision problem, as specified by its owner, is **instrument gating, not target
selection**. EarthCARE carries a passive multi-spectral imager (MSI) alongside an active 94 GHz
Cloud Profiling Radar (CPR) (Illingworth et al., 2015; Wehr et al., 2023). In the project's
abstraction, MSI runs continuously and cheaply across its 150 km swath, while the CPR is the
expensive, power-limited instrument. The agent observes MSI-measurable quantities along a pass and
decides **whether to power up the CPR for that pass**; correctness is scored afterwards against the
model-truth precipitation fields (`prectot`, `preccon`).

*(This is a design abstraction adopted for the RL study. It is not a description of EarthCARE
flight operations, in which the CPR is not duty-cycled in this manner. The abstraction is a
legitimate and common way to pose an instrument-tasking problem; it is noted here only so that the
critique is not mistaken for a claim about the real mission.)*

Two consequences follow immediately, and they overturn parts of an earlier reading of this
notebook.

**`spaces.Discrete(2)` is correct design, not a defect.** The decision genuinely is binary — CPR on
or CPR off for the pass. A fixed action space of cardinality two is the faithful encoding of that
decision, not a workaround for variable target counts.

**The absence of `cnprcp_mean` / `prectot` / `preccon` from `_get_obs` is correct, not a defect.**
These fields are the *label*. Placing them in the observation would be textbook target leakage
(Kapoor & Narayanan, 2023): the agent would achieve near-perfect scores by reading the answer, and
the resulting policy would be unrunnable in flight, where the truth field does not exist. The
correct requirement is not that the reward variable be observable — it must not be — but that the
observable features be *predictive* of it. That converts an alleged defect into an empirical
question, which §7.5 answers with measurement.

### 7.3 Retraction: "prioritization is not expressible" — withdrawn

An earlier version of this section argued as follows: because cell 13 of the data notebook
aggregates every cluster intersecting a pass into a single row via `sum`/`mean`/`max`, and because
the environment then exposes only `spaces.Discrete(2)`, the question "*which* target should be
prioritized?" is not expressible, and therefore "if the project's goal is target prioritization,
the current formulation cannot express the goal."

**That conclusion is withdrawn.** It critiques a goal the project does not hold. Under the CPR-gating
formulation of §7.2 there is nothing to prioritize among: the pass is the decision unit, the
aggregation in cell 13 is the appropriate summarization of what the pass contains, and a binary
action is the correct action space. The premise of the argument (aggregation destroys per-target
identity) is factually correct and is retained in §5.3 where it bears on object-level statistics;
the *inference* drawn from it about the RL formulation was wrong.

The related observation about variable cardinality is likewise demoted rather than deleted. It
remains true that connected-component labelling yields a different number of clusters each hour —
cells 21 and 23 record `min 1`, `max 14`, `mean 3.502` observed clusters per frame, with labels
renumbered from scratch every frame and no identity across timesteps. Under CPR gating this is not
an action-space problem. It is a *feature-construction* problem: the per-pass covariates
(`count`, `area`, `tot_prectot`, `avg_prectot`, `max_prectot`) are computed by aggregating over an
object set whose membership is an artefact of the 220 K threshold (§5) and whose cardinality
fluctuates for reasons unrelated to the atmosphere. The machinery for variable action sets
(Chandak et al., 2020a, 2020b; Dulac-Arnold et al., 2015) is not needed here and the earlier
citation of it in this context was misdirected; it would become relevant only if the project later
posed a multi-target selection problem.

### 7.4 What survives: defects that are present in the implemented environment

**(a) Transitions are action-independent — the gating problem is not actually a scheduling problem.**
This is the most consequential finding in this section, and the CPR framing strengthens rather than
weakens it.

```python
def step(self, action):
    reward = self._compute_reward(action)
    self._index += 1
    done = (self._index >= self.n_steps - 1)
    obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
    ...
```

`self._index` advances unconditionally. No action influences any future observation or reward. The
state sequence is a fixed replay of the dataframe rows.

Under target prioritization this would merely be a modelling simplification. Under **CPR gating it
is a modelling error**, because the entire point of gating a power-limited instrument is that
switching it on *consumes something*: battery state of charge, thermal margin, on-board storage,
downlink volume, duty-cycle allowance. Those resources are precisely the state variables that make
instrument tasking sequential, and they are exactly what the agent must learn to husband. A correct
formulation carries at least one such resource in the state and lets the action decrement it. Here,
none exists: turning the CPR on is free, so the agent faces no budget, no opportunity cost, and no
reason to defer. This is the standard structure in the satellite-tasking MDP literature, where
power and data-volume state are explicit (Eddy & Kochenderfer, 2020; Herrmann & Schaub, 2023a,
2023b).

Two things follow.

*First, the problem degenerates to a contextual bandit.* With action-independent transitions the
optimal policy maximizes each immediate reward independently; discounting is vacuous and
temporal-difference bootstrapping propagates no information (Sutton & Barto, 2018, Ch. 2 vs. Ch. 3).
A2C, DQN, and C51 converge to the same greedy per-row rule that a calibrated classifier with a
decision threshold reaches directly, but with higher variance and no interpretable value function.
Distributional RL (C51) applied to a bandit with a deterministic, action-independent state sequence
is machinery without a corresponding problem. Any comparison among these algorithms measures
optimizer noise, not scheduling competence.

*Second, the hand-tuned reward silently sets the duty cycle.* Because the transition function
encodes no resource cost, the entire cost of firing the CPR is carried by the false-positive
penalty in `_compute_reward`:

```python
if action == 1:
    if cnprcp_mean > 0:
        reward = 1 + (cnprcp_mean * scale)   # scale = 10000
    else:
        reward = -0.1
else:
    reward = -0.001 if cnprcp_mean > 0 else 0
```

Let $p$ be the agent's belief that the pass is precipitating and $b = \texttt{scale}\times
\mathbb{E}[\texttt{cnprcp\_mean}\mid \texttt{>0}]$ the expected intensity bonus. Acting is preferred
when

$$p(1+b) - (1-p)(0.1) \;>\; -0.001p
\;\Longleftrightarrow\; p\,(1 + b + 0.101) > 0.1
\;\Longleftrightarrow\; p > \frac{0.1}{1.101 + b}.$$

With no intensity bonus ($b=0$) the break-even belief is $0.1/1.101 = 9.1\%$. For a
$1\ \text{mm hr}^{-1}$ cell, $1\ \text{mm hr}^{-1} = 2.78\times10^{-4}\ \text{kg m}^{-2}\text{s}^{-1}$,
so $b = 10^{4} \times 2.78\times10^{-4} = 2.78$ and the break-even belief falls to
$0.1/3.881 = 2.6\%$. *(Units of `cnprcp_mean` in the 4-month table were not verified; the
calculation assumes the `kg m⁻² s⁻¹` convention that the 0.5° collection uses per its DAS.)*

A break-even belief of 2.6–9.1% is an extremely permissive gate. Combined with the label definition
of §7.7 — under which 99.84% of rows are "precipitating" — the induced optimal policy is **fire the
CPR on essentially every pass**. The duty cycle that the agent is nominally being trained to manage
is therefore fixed at ≈ 100% by the ratio of two hand-chosen constants (`1` and `-0.1`), not by any
power budget. Two hyperparameters, neither traceable to a physical quantity, determine the answer.

*Remediation:* add a resource variable to the state (e.g. remaining CPR-on seconds in a rolling
orbit budget), decrement it in `step()` when `action == 1`, terminate or heavily penalize on
exhaustion, and remove the ad-hoc `-0.1`. This makes the transition action-dependent, restores a
genuine MDP, and makes the operating point a consequence of the budget rather than of reward
tuning.

**(b) Reframed: the observation set must be shown to be predictive.** The earlier finding under this
heading — "the reward-determining variable is not in the observation" — is reframed rather than
retracted. Its factual content stands: the observation is

```python
self.obs_dim = 4 + n_one_hot   # x_norm, y_norm, z_norm, ground_track, + one-hot(time_range)
```

i.e. position (Cartesian unit vector, min-max scaled), a `ground_track` flag, and a one-hot time
bin, while the reward depends on `cnprcp_mean`. Under §7.2 this separation is *required*. What is
defective is not the separation but the *choice of observables*: the implemented observation
contains **no MSI-measurable radiometric quantity at all**. It is geometry and clock only. The agent
is asked to predict precipitation from where and when the satellite is, which is a climatology, not
a measurement.

This matters because the pipeline already computes the relevant observables. `tautot` (total cloud
optical thickness) is fetched and attached to every ground-track row in cell 6 of the data notebook,
and is then used only for the decorative KDE figures of cells 18–19. It never reaches the
environment. §7.5 shows this is the single most informative feature available.

Reported F1 differences between DQN, A2C, and C51 trained on the geometry-only observation are
therefore unlikely to be attributable to the algorithms; they are bounded by the mutual information
between position/time-bin and precipitation.

**(c) The `ground_track` feature is almost certainly constant.**

```python
world = gpd.read_file('110m_cultural.zip', layer = 'ne_110m_admin_0_boundary_lines_land')
geometry = [Point(xy) for xy in zip(data['lon_sat'], data['lat_sat'])]
geo_full['ground_track'] = geo_full.apply(lambda row: 0 if world.contains(row.geometry).any() else 1, axis=1)
```

`ne_110m_admin_0_boundary_lines_land` is a **line** layer, not a polygon layer. Under GEOS
semantics a `LineString` contains a `Point` only if the point lies in the line's interior — a
measure-zero condition for arbitrary satellite coordinates. `ground_track` is therefore expected to
be identically 1, contributing a constant to the observation vector. Of the four continuous
observation dimensions, one is thus expected to be uninformative by construction. *This is reasoned
from the layer's geometry type; I did not execute the cell to confirm, and the recorded outputs do
not report the column's value counts.*

### 7.5 Measured learnability of the gating decision

The question raised in §7.2 — are MSI-observable features predictive of the label? — has been
answered empirically since this document was first drafted. The measurement was made over
**2,270,592 cell-hours** of the 0.5°-derived tessellation dataset (consistent with a
$73 \times 144 = 10{,}512$-cell 2.5° grid over $216$ hourly frames: $10{,}512 \times 216 =
2{,}270{,}592$). *These figures were supplied by the project owner; I did not run the computation
and report them as received.*

Single-feature discrimination, ROC-AUC against three label definitions:

| Feature | `prectot ≥ 0.1 mm/hr` | `prectot ≥ 1 mm/hr` | `preccon ≥ 0.1 mm/hr` |
|---|---|---|---|
| `tautot` (total cloud optical thickness) | **0.883** | **0.887** | **0.926** |
| `cre` = `lwtupclr` − `lwtup` | 0.785 | 0.866 | 0.773 |
| `cldtot` (total cloud fraction) | 0.731 | 0.811 | 0.585 |
| `cldhgh` (high-cloud fraction) | 0.638 | 0.726 | 0.730 |
| `cldtmp` (cloud-top temperature) | 0.333 | 0.252 | 0.222 |
| `lwtup` (TOA upwelling LW — *the field the detector uses*) | 0.367 | 0.263 | 0.435 |

An AUC below 0.5 indicates inverse prediction, which is physically expected for `cldtmp` and
`lwtup`: colder cloud tops and lower outgoing longwave flux accompany wetter columns. Reflecting
those rows about 0.5 gives sign-corrected discrimination:

| Feature (sign-corrected) | `prectot ≥ 0.1` | `prectot ≥ 1` | `preccon ≥ 0.1` |
|---|---|---|---|
| `cldtmp` | 0.667 | 0.748 | 0.778 |
| `lwtup` | 0.633 | 0.737 | 0.565 |

A logistic model on all six features reaches **ROC-AUC 0.935** for `prectot ≥ 1 mm/hr` against a
**1.94%** base rate, with average precision **0.163** — a lift of

$$\frac{0.163}{0.0194} = 8.40\times$$

over the base rate. Against `preccon > 0` the same model reaches ROC-AUC **0.870**. (With a base
rate this low, average precision is the more informative summary and ROC-AUC alone is optimistic;
Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015.)

Three conclusions follow.

**The gating decision is learnable.** An 8.4× lift over a 1.94% base rate from six passive
radiometric fields establishes that a CPR-gating policy conditioned on MSI-observable quantities has
real signal to exploit. The formulation of §7.2 is sound; §7.4(b)'s complaint is about which
observables were wired in, not about whether any exist.

**The field the detector thresholds is the weakest of the six.** For convective precipitation,
`lwtup` scores 0.565 sign-corrected — barely above chance — while `tautot` scores 0.926. This is an
independent, quantitative confirmation of Defect 1 (§3) obtained without any appeal to latitude:
the pipeline built its entire target definition on the least informative available field. It is also
a quantitative confirmation of Defect 6 (§10, R3): the strongest discriminator,
`tautot`, is already downloaded by cell 6 and discarded.

**The CRE remediation (R1) is validated but is not the best available option.** `cre` improves on
raw `lwtup` by $0.773 - 0.565 = 0.208$ AUC for convective precipitation — a large gain, confirming
the physical argument of §10, R1. But `tautot` alone beats it by a further $0.926 - 0.773 = 0.153$.
R1 should therefore be reordered: use `tautot` as the primary discriminator, `cre` as the
physically-motivated secondary, and treat the pair as complementary (optical thickness is a cloud
property; CRE is a radiative one, and the near-zero CRE of clear cold surfaces remains the cleanest
available *exclusion* test for the polar false positives of §3).

*Caveats, stated by the project owner and retained here.* (i) These statistics are **per grid cell,
not per pass**; a pass-level gating decision aggregates over ~100 cells along the swath, which will
change both the base rate and the achievable AUC, in general favourably for the aggregate. (ii) The
sample is a **9-day window in May 2005**, so it does not escape the seasonal confound of §6.2 —
it merely reduces it from one day to nine consecutive days in the same season. Cross-seasonal
validation remains outstanding, and §6's remediation R6/R7 is a precondition for it. (iii) The
labels are model-truth fields from the same simulation that produced the features, so these numbers
bound learnability *within the OSSE*, not transferability to real MSI radiances.

### 7.6 The coupling to the 0.5° table is prospective

`A2C_Model.ipynb` reads `zip://clustered_data_4months.zip!clustered_data_4months (2).geojson` with
columns `cnprcp_mean`, `lat_sat`, `lon_sat`, `time_range` — a different, earlier table, not the
0.5° pipeline's `observedclusters_8months.geojson`. Defects 1–4 therefore do not yet contaminate
the RL results in this notebook; they will when the tables are joined. In particular, §6.3's
conclusion applies immediately on substitution: with a 24-hour replication cycle, the
`TimeSeriesSplit(n_splits=3)` used in cell 8 would place all 24 atmospheric states in every fold.

### 7.7 A label with almost no entropy

Under CPR gating the label definition is not a detail: it *is* the definition of a correct
decision. The current definition is degenerate.

Cell 26 of the data notebook records `1351910` of `1354115` cluster rows with `avg_prectot > 0`, so
$p = 0.998372$ and $q = 2205/1{,}354{,}115 = 0.0016284$. The binary entropy of a
"precipitating vs. not" label defined as `> 0` is

$$H = -p\log_2 p - q\log_2 q = 0.998372(0.002353) + 0.0016284(9.2626) = 0.0174\ \text{bits}.$$

`A2C_Model.ipynb` defines its reward and its evaluation label exactly this way
(`if cnprcp_mean > 0`). A label carrying 0.017 bits cannot support a meaningful F1 score, and — via
the break-even calculation of §7.4(a) — it forces the CPR duty cycle to ≈ 100%, which is the
degenerate answer to a gating problem.

The measurements of §7.5 use the correct alternative: a **magnitude** threshold. At
`prectot ≥ 1 mm/hr` the base rate is 1.94%, giving a binary entropy of

$$H = -0.9806\log_2 0.9806 - 0.0194\log_2 0.0194 = 0.0275 + 0.1104 = 0.138\ \text{bits}$$

— roughly eight times the information content of the `> 0` label, and a base rate at which a gating
decision is a genuine trade-off rather than a foregone conclusion. Any future reward function should
be defined against a magnitude threshold or a quantile, not against `> 0`.

---

## 8. Additional defects observed

These are secondary but independently checkable.

### 8.1 Grid-cell polygon convention is inconsistent with the area computation

Cell 8 builds geometry as `box(lo, la, lo + grid_size, la + grid_size)` — treating the G5NR
coordinate as the cell's **south-west corner**. Cell 7 computes area as

```python
area[i, :] = (R**2) * dlon_rad * (np.sin(lat_rad[i] + dlat_rad / 2) - np.sin(lat_rad[i] - dlat_rad / 2))
```

— treating the same coordinate as the cell **centre**. Under the centre convention (the standard one
for a 361-point axis spanning −90…90 at 0.5°), every polygon is displaced by +0.25° in latitude and
+0.25° in longitude, i.e.

$$0.25^\circ \times \frac{\pi R}{180} = 27.8\ \text{km meridionally}, \qquad 27.8\cos\phi\ \text{km zonally}.$$

That is **18.5% of the 150 km MSI swath width**, and it biases the `observed` intersection test. It
also explains why the southernmost observed centroid is −74.75 rather than −75.00.

### 8.2 No longitudinal periodicity in the labelling

`ndi.label` operates on a plain 2-D array with no wraparound. Any cold-cloud system crossing the
±180° meridian is split into two components at the array seam, inflating the object count and
truncating precisely the largest systems. (The repository's tessellation notebook addresses this
with an explicit `% n_lon`.)

### 8.3 Diagnostic figures are built from a biased head slice

Cells 15–16 use `clusters_filtered[:1100]` after `clusters['time'].dt.day == 19`. Two problems:
`dt.day == 19` selects the 19th of *every* month, not one month (the figure titles say "1 month
period"); and `[:1100]` takes the first 1,100 rows of a table sorted by `(time, cluster)`. Because
`ndi.label` numbers components in raster-scan order beginning at the first array row — which, after
`lat=slice(-75, 75)`, is −75° — the lowest cluster indices are systematically the southernmost
features. The diurnal-cycle figure is therefore built from an Antarctic-biased subsample of a single
atmospheric state.

### 8.4 Solar hour is offset from the field it labels

Cell 9 evaluates solar hour at `r.time + frame_duration/2`, but `frame_duration` was last bound in
cell 3 to `batch_duration` (10 minutes), not `g5nr_frame_duration` (1 hour). Solar hour is therefore
computed at $t + 5$ min for an hourly-mean field whose centre is $t + 30$ min — a systematic 25-minute
(0.42 h) offset. Minor relative to Defects 1–4, but it propagates into cells 13, 15, and 16.

### 8.5 Nearest-neighbour time selection adds a further offset

`_fetch_chunk` requests `HH:04` and applies `.sel(time=pick, method="nearest")`. The G5NR hourly
stamps sit at `:30` (per the DAS span recorded in the sibling notebook). Since
$|HH{:}04 - (HH{-}1){:}30| = 34$ min and $|HH{:}04 - HH{:}30| = 26$ min, requests snap forward to
`HH:30`. The count of distinct slices is unaffected (still 24), but the row's nominal hour label is
26 minutes earlier than the field it carries.

---

## 9. Cross-cutting discussion: how the defects compound

The defects are not additive; they multiply, and the notebook's own numbers show the product.

**Defect 1 × Defect 4 (detector × frozen season).** The detector's false-positive rate is a strong
function of season, because the clear-sky emission temperature of high-latitude surfaces is. Freezing
the atmosphere at 20 May samples that rate at its southern-hemisphere maximum. The result is the
11.0× hemispheric asymmetry of §6.2 — an artefact that would be diagnosed immediately if the pipeline
walked the real time axis, because the asymmetry would migrate with the season and the polar
detections would visibly follow the winter hemisphere. With a frozen date, it looks like a stable
property of the world.

**Defect 1 × Defect 3 (detector × dissolve).** A false positive from a cold *surface* is
qualitatively different from a false positive from a cold *cloud*: surfaces are spatially coherent
over continental scales, so thresholding an ice sheet produces one enormous connected component
rather than many small ones. The dissolve then converts spatial extent into a single row with
maximal `area` and near-zero `avg_prectot`. Under Defect 1 alone, the polar false positives would be
numerous but individually small; the dissolve concentrates them into few, enormous, zero-reward
objects — which is worse for learning, because a large number of small errors averages out while a
small number of large ones does not.

**Defect 3 × Defect 3′ (dissolve × unweighted crediting).** Because the giant objects nearly always
intersect the swath (hit probability ≈ 1 vs. 0.765% for a point), they enter the positive class
almost every hour; and because cell 13 credits whole-cluster `area` and `tot_prectot` to any pass
that touches them, each such pass is labelled with up to 39× more observed area than the instrument
can image. Selection bias and feature inflation therefore act on the same rows.

**Defect 2 as an amplifier of the appearance of correctness.** The latitude clip removes the
*visibly* absurd detections (the South Pole itself) while retaining the bulk of the contaminated
area, and it makes the cell-24 diagnostic report `0.0%` in both polar bins. The pipeline thus emits
a diagnostic that appears to exonerate it. This is the most instructive interaction in the audit:
a mitigation that is ineffective but *appears* effective is more dangerous than no mitigation.

**Defect 4 × Defect 5 (replication × evaluation).** The 245-fold replication has a 24-hour period,
so it defeats every temporal-splitting strategy simultaneously. Any evaluation of the RL agent (or
of the `RandomForestClassifier` baseline) on this table would report performance on atmospheric
states it had already memorized. Because the replication is exact, the leak is total rather than
partial.

**Net effect on the reward signal.** The intended reward is "precipitation observed". What the
pipeline delivers is: a positive class enriched 6.1× toward a 5%-of-Earth polar band, whose largest
members carry ≈ 0 precipitation but maximal area, drawn from 24 atmospheric states of a single
autumn day, with a binary "precipitating" label carrying 0.017 bits, credited to passes without
intersection weighting. An agent trained on this will learn a well-defined function — it will simply
not be "fire the CPR when there is convection to profile".

---

## 10. Recommended remediations

Each is tied to the defect it addresses, with its cost stated.

### R1 — Replace the OLR threshold with `tautot` and the cloud longwave radiative effect (→ D1, D6)

**Measured ordering (see §7.5).** Against `preccon ≥ 0.1 mm/hr`, single-feature ROC-AUC is
`tautot` 0.926 > `cre` 0.773 > `lwtup` 0.565 (sign-corrected). The primary discriminator should
therefore be `tautot`, which the pipeline already downloads in cell 6 and currently discards; CRE
is the physically-motivated secondary. The physical argument for CRE below stands and is confirmed
by the measurement (+0.208 AUC over raw `lwtup`), but it should not be adopted *instead of*
`tautot`.

The collection exposes `lwtupclr`, the clear-sky TOA upwelling longwave flux computed by the model's
own radiation scheme for the same column. Define

$$\mathrm{CRE}_{\mathrm{LW}} = \texttt{lwtupclr} - \texttt{lwtup},$$

the longwave cloud radiative effect in the sense of Ramanathan et al. (1989). This is the *difference*
the current detector cannot express. Over a clear cold surface — plateau, sea ice, winter continent —
the two fields coincide by construction and $\mathrm{CRE}_{\mathrm{LW}} \approx 0$, regardless of how
cold the surface is. Under an optically thick anvil the difference is large: inverting
Stefan–Boltzmann, a clear tropical column at $T_b \approx 265$ K emits ≈ 280 W m⁻² and an anvil at
$T_b \approx 205$ K emits ≈ 100 W m⁻², giving $\mathrm{CRE}_{\mathrm{LW}} \approx 180$ W m⁻². Both
columns have `lwtup` ≈ 100–117 W m⁻² and are *indistinguishable* under the current detector; they
differ by ~180 W m⁻² under CRE.

*Cost:* one additional variable per timestep (2× I/O for the detector fields; 3× with `tautot`).
*Caveat:* neither `tautot` nor CRE identifies convection per se — both identify optically thick
high cloud — so both should be confirmed dynamically (R3). CRE retains one property `tautot` does
not: its near-zero value over clear cold surfaces makes it the cleanest available *exclusion* test
for the polar false positives of §3. *The illustrative flux values above are my Stefan–Boltzmann inversions of plausible emission
temperatures, not measurements from this dataset; the authoritative numbers are `lwtupclr` and
`lwtup` themselves, which the pipeline can compute directly.*

### R2 — Use `cldtmp` instead of inverting broadband OLR — **tested, partially successful** (→ D1)

`cldtmp` is the model's diagnosed cloud-top temperature — the quantity the 220 K threshold was
attempting to approximate. The collection's DAS has since been read and confirms the field's
identity: `long_name "cloud_top_temperature"`, `units "K"`, with
`_FillValue = missing_value = 1.0E15`.

**Measured outcome.** Substituting `cldtmp < 220 K` for the broadband detector was tested by the
project owner on the tessellation dataset. *(Figures supplied by the project owner; I did not run the
computation.)*

| Quantity | Old detector `(lwtup/σ)^{1/4} < 220 K` | `cldtmp < 220 K` |
|---|---|---|
| High-latitude cells flagged | 53.91% | 7.63% (86% fewer detections) |
| Tropical cells flagged | 1.06% | 20.64% |
| Of tropical flagged, with no convective precipitation | 3.48% | 11.71% |

The substitution is a large improvement at high latitude and restores a plausible tropical detection
rate, which the broadband detector had suppressed to 1.06%. But two caveats emerged.

*`cldtmp` is not a clean cloud mask.* It carries a finite value over **18.3%** of cells with
`cldtot < 0.01`, so a fill-value test alone does not isolate cloudy columns. Gating on `cldtot > 0.5`
removed only 5.8% of detections overall and 0.7% of the polar ones — i.e. the residual polar
detections are not an artefact of spurious `cldtmp` values in clear columns.

*The residual detections are real cold cloud, not convection.* The honest reading of the 7.63% polar
figure is that `cldtmp < 220 K` detects cold **cloud**, which at high latitude genuinely exists and
genuinely precipitates — as stratiform snow, not convection. R2 therefore fixes the *surface*
contamination of Defect 1 but not the cloud-type ambiguity. It must be combined with a convective
discriminator (R3: `preccon`, `cape`) or with the snow/rain partition (`precsno`) if the target class
is meant to be convective.

*Cost:* one additional variable; explicit fill-value masking at 1.0E15; and, because R2 alone does
not identify convection, at least one further field from R3.

### R3 — Add physical confirmation from `preccon`, `cape`, `cldhgh`, `tauhgh` (→ D1, D6)

All are present in the DDS (verified). Their physical bases as convection discriminators:

| Variable | Basis |
|---|---|
| `preccon` | Convective precipitation from the model's convection parameterization — nonzero **only** where the scheme is active. A cold clear ice sheet has `preccon` ≡ 0. This is the single most direct label for "convection is occurring" available in the collection. |
| `cape` | Convective available potential energy: the thermodynamic precondition. High CAPE with cold cloud tops distinguishes convective from synoptic/frontal cold cloud. Near-zero over the polar plateau by construction. |
| `cldhgh` | High-cloud fraction. Deep convection produces near-unity high-cloud fraction; clear polar surfaces produce zero. Separates "the column is cold" from "the column is cold *because of high cloud*". |
| `tauhgh` | High-cloud optical thickness. Deep convective cores are optically thick ($\tau \gtrsim 20$); thin cirrus and clear sky are not. Distinguishes an active core from a decaying anvil, which matters because a 220 K threshold treats both identically. |
| `cldtot`, `cldmid`, `cldlow` | Vertical structure of cloudiness; permit an explicit deep-vs-shallow test. |
| `precsno` | Snow vs. rain partition. High-latitude precipitation is predominantly snow; the split provides a second, independent high-latitude discriminant and is directly relevant to what MSI can usefully observe. |
| `lwtupclr` | See R1. Also `lwtupclrcln` (clear-sky, aerosol-free) is available if aerosol effects need to be isolated. |

*Cost:* the multi-field detector needs 4–7 variables per timestep instead of 1. See R6 for how to
pay for it.

### R4 — Replace connected-component objects with a fixed tessellation, or use hierarchical segmentation (→ D3, D5)

Two options with different trade-offs:

**(a) Fixed equal-angle (or equal-area) tessellation.** Every cell is a persistent target with a
stable identity across time; the target count is constant; `observed` reduces to integer cell
binning, eliminating both the `unary_union` cost and the size-bias of §5.2; and the RL action space
becomes well-defined. *Trade-off:* a "target" becomes a tile rather than a storm, so object-level
semantics (system lifetime, growth rate) are lost. **This option is already prototyped in the
repository** as `geos5data_0.5deg_tessellation_8mo.ipynb`, which uses a 2.5° grid
(`STRIDE = 5`), walks the real hourly axis, computes `cre = lwtupclr - lwtup`, and defines the
positive class by quantile. *I did not verify whether it has been run to completion; its
`SMOKE_TEST` flag is set to `True` in the working tree.*

**(b) Hierarchical / space–time segmentation.** TOOCAN (Fiolleau & Roca, 2013, 2024) and PyFLEXTRKR
(Feng et al., 2023) both resolve the merging problem by segmenting in a 3-D space–time domain with
nested thresholds rather than a single cut, and both provide object identity across timesteps.
*Trade-off:* substantially more complexity and its own tuning surface; Feng et al. (2025) document
how much resulting statistics still vary between algorithms.

### R5 — If objects are retained, weight features by intersection and expose area (→ D3)

Compute per-pass `area`, `count`, and `tot_prectot` over the **swath ∩ cluster** intersection rather
than the whole cluster (§5.2). Additionally, report `observed` rates conditioned on cluster area so
the size bias is visible in the diagnostics, and normalize any area-based reward term explicitly.

*Cost:* geometric intersection per (pass, cluster) pair instead of a boolean predicate — more
expensive, but bounded by the same join.

### R6 — Walk the real G5NR time axis (→ D4)

Replace `_frame_ds_time` with the sibling notebook's implementation, verbatim:

```python
def _frame_ds_time(frame):
    return g5nr_start + min(int(frame), n_g5nr - 1) * g5nr_frame_duration
```

and apply the same correction to cell 6's `lookup_tautot`.

*Cost, quantified.* The fetch scales 245× (24 slabs → 5,880); CPU work is unchanged. Transfer volume
for a full-globe fetch of two variables over 5,880 hours is
$361 \times 720 \times 4 \times 2 \times 5880 = 12.23$ GB. Two mitigations, both already present in
the repository: (i) the sibling notebook's 24-hour chunking with checkpointing and handle-reopening
retries makes the run resumable; (ii) server-side striding, as in the tessellation notebook
(`isel(lat=slice(None,None,5), lon=slice(None,None,5))`), reduces a 7-variable fetch to

$$\lceil 361/5\rceil \times \lceil 720/5\rceil \times 4 \times 7 \times 5880 = 73 \times 144 \times 4 \times 7 \times 5880 = 1.73\ \text{GB},$$

i.e. **more variables for one-seventh the volume**, at the price of a 2.5° target grid.

### R7 — If the full axis is infeasible, sample multiple dates across the seasonal cycle (→ D4)

Four dates near the solstices and equinoxes cost 4× the current I/O rather than 245×, and break the
season confound of §6.2 — the dominant source of the 11.0× hemispheric asymmetry. This is a strictly
inferior but cheap intermediate. *Trade-off:* still not a continuous record, so no MCS lifecycle or
temporal-autocorrelation analysis is possible.

### R8 — Make the transition action-dependent, and wire in the observables that exist (→ D5)

The decision problem is settled (CPR gating, §7.2); the environment does not yet implement it.
Three changes, in order of importance.

**(i) Give the action a cost in the transition function (§7.4a).** Carry at least one resource in
the state — remaining CPR-on seconds in a rolling orbit or daily budget is the minimal choice —
decrement it in `step()` when `action == 1`, and terminate or penalize on exhaustion. Remove the
ad-hoc `-0.1` false-positive penalty, whose only current function is to stand in for the missing
budget. *Trade-off:* the environment can no longer be a straight replay of a dataframe, since the
resource state must persist across steps; episode boundaries must be defined (e.g. one orbit, or one
day). This is the change that turns a bandit back into a scheduling MDP and makes the operating
point a consequence of the power budget rather than of two hand-tuned constants. Explicit
power/data-volume state is standard in the satellite-tasking MDP literature (Eddy & Kochenderfer,
2020; Herrmann & Schaub, 2023a, 2023b).

**(ii) Put MSI-observable radiometry in the observation (§7.4b).** The current observation is
geometry and clock only. `tautot` is already attached to every ground-track row in cell 6 of the
data notebook and is the strongest single predictor measured (§7.5, ROC-AUC 0.926 against
`preccon ≥ 0.1 mm/hr`); `cre`, `cldtot`, `cldhgh` follow. Adding them is a data-plumbing change, not
a research problem. The label (`prectot`/`preccon`) must stay out of the observation — that
separation is correct and must be preserved (Kapoor & Narayanan, 2023).

**(iii) Define the positive class by magnitude (§7.7).** Replace `> 0` (0.017 bits) with a magnitude
or quantile threshold; `prectot ≥ 1 mm/hr` gives a 1.94% base rate and 0.138 bits.

**Baseline discipline.** Until (i) is implemented the problem is a contextual bandit, so a
calibrated classifier with a tuned decision threshold is the correct baseline, and any advantage
claimed for A2C/DQN/C51 over it should be demonstrated rather than assumed (Sutton & Barto, 2018).
Because the base rate is ~2%, report average precision alongside ROC-AUC (Davis & Goadrich, 2006;
Saito & Rehmsmeier, 2015).

### R9 — Do not evaluate on a replicated table (→ D4, D5)

Until R6 is applied, no temporal holdout on the 0.5° table is clean (§6.3). After R6, hold out by
*distinct G5NR timestamp*, with a buffer of at least the decorrelation timescale of the field
(hours to a day for convection), following the blocked-cross-validation guidance of Roberts et al.
(2017).

### R10 — Secondary corrections (→ §8)

Fix the half-cell polygon offset (§8.1) by using `box(lo - grid_size/2, la - grid_size/2, lo + grid_size/2, la + grid_size/2)`
or by consistently adopting the corner convention in `compute_grid_cell_area`; add longitudinal
wraparound to the labelling (§8.2); bind `frame_duration` explicitly where solar hour is computed
(§8.4); and replace the head-slice diagnostics with random or stratified samples (§8.3).

---

## 11. Threats to validity of this critique

1. **The notebook was not re-executed.** Every quantitative claim is either arithmetic I performed
   independently (shown in-line) or a value read from the notebook's stored cell outputs. If cells
   were edited after execution, those outputs are stale relative to the source shown in §2.2. The
   internal consistency of the recorded numbers ($5{,}527 \times 245 = 1{,}354{,}115$;
   $5{,}508 \times 3.502 = 19{,}289 \approx 19{,}290$; $5{,}880 \times 6 = 35{,}280$) argues that
   they correspond to the source as shown, but this is inference, not verification.

2. **Six of seven RL/classifier notebooks could not be read** (§7.1). The claim that the project's
   RL algorithms assume a fixed action space is verified only for `A2C_Model.ipynb`. The other
   notebooks may pose the problem differently, including in ways that would restore the original
   variable-cardinality framing.

3. **GEOS-5 variable semantics — partially resolved since drafting.** The 71 variable names and
   their `[18288][361][720]` shapes are verified from the DDS. The collection's DAS has since been
   retrieved and confirms the fields this critique relies on:
   `lwtup` = `upwelling_longwave_flux_at_toa` (W m⁻²) — a **flux**, not a radiance, which is the
   documentary basis for §3.1; `lwtupclr` = `upwelling_longwave_flux_at_toa_assuming_clear_sky`;
   `cldtmp` = `cloud_top_temperature` (K); `preccon` = `convective_precipitation` (kg m⁻² s⁻¹);
   `prectot` = `total_precipitation` (kg m⁻² s⁻¹); `precsno` = `snowfall` (kg m⁻² s⁻¹);
   `cldhgh` = `cloud_area_fraction_for_high_clouds`; `cape` = `cape_for_surface_parcel`, declared with
   `units "J m-2"` (note: CAPE is conventionally J kg⁻¹, so this declared unit should be checked
   against the GEOS-5 file specification before `cape` is used quantitatively). All fields carry
   `_FillValue = missing_value = 1.0E15`. What the DAS does **not** state is `cldtmp`'s clear-column
   convention; the empirical test reported in R2 shows it carries a finite value over 18.3% of cells
   with `cldtot < 0.01`, so it is not a pure cloud mask, and a fill-value test alone is insufficient.
   The weighting used to define `cldtmp` remains unverified.

4. **Illustrative flux values are inversions, not measurements.** The tropical-anvil and
   Antarctic-plateau OLR values in §3.2 and R1 are Stefan–Boltzmann inversions of plausible emission
   temperatures, presented to make the overlap concrete. The decisive evidence for Defect 1 is not
   these numbers but the notebook's own latitude histogram (30.5% of detections in 5.0% of the
   surface). A definitive quantification would compute the joint distribution of `lwtup` and
   `preccon` directly from the archive — which the tessellation notebook's cell 10 is set up to do
   and which I did not run.

5. **The polar detections are not claimed to be *entirely* spurious.** Antarctic coastal and
   Southern Ocean frontal systems do produce genuine cold cloud with real precipitation. The claim
   is that the detector cannot distinguish these from clear cold surfaces, evidenced by the 6.1×
   area-relative enrichment and the 11.0× hemispheric asymmetry — not that every high-latitude
   detection is false. Establishing the false-positive *rate* requires `preccon`.

6. **The magnitude of the giant-cluster pathology is smaller than initially hypothesized.** The
   hypothesis under review posited single polar components of order $10^7$ km². The notebook's
   printed head shows the largest visible component in frame 0 at 1,170 cells ≈ $9.9\times10^{5}$
   km² — an order of magnitude smaller, though still 10–100× a large MCS. I saw only the first and
   last five rows of the table (pandas repr truncation), so this is a lower bound, not the maximum;
   the theoretical ceiling for a band-spanning component is $2.55\times10^{7}$ km². The mechanism is
   confirmed; the stated magnitude is corrected.

7. **An earlier version of this document mis-stated the project's decision problem, and §7 was
   revised accordingly.** That draft assumed the goal was target prioritization and concluded that
   the implemented formulation could not express it. The goal is CPR gating (§7.2); the conclusion is
   retracted in §7.3. Readers holding the earlier version should replace §7 in full. The revision
   changes the *location* of Defect 5, not its severity: the action-independent transition function
   (§7.4a) is a more serious error under gating than it would have been under prioritization, because
   gating a power-limited instrument is meaningful only if firing it consumes a budget.

   The related severity caveat for Defect 3 stands: under the per-pass binary formulation the
   giant-cluster pathology is attenuated (though not removed — such clusters still enter as
   high-`area`, near-zero-`prectot` positives, and inflate the per-pass `area` and `tot_prectot`
   covariates without intersection weighting).

   A further caveat now applies to §7.5. Those learnability figures were supplied by the project
   owner; I did not run the computation, and I have not seen the fitting code, the train/test
   protocol, or whether the logistic model's reported AUC is in-sample. They are reported as
   received, and the direction of their conclusions (that the task is learnable, and that `tautot`
   dominates `lwtup`) is consistent with the physical argument of §3 and §10-R1 — but that
   consistency is not independent verification.

8. **Defect 2 is the most arguable.** Latitude clipping is not incorrect; it is ineffective for its
   stated purpose. It is presented as a symptom of Defect 1 and as a cautionary case of a mitigation
   that suppresses its own diagnostic, not as an independent methodological error.

9. **`ground_track` (§7.3c) is reasoned, not executed.** The conclusion follows from the layer being
   a line geometry, but I did not run the cell or inspect its output.

10. **This critique does not evaluate the science question itself** — whether deep convection is the
    right target class for CPR gating, whether a duty-cycled CPR is a realistic operational premise, or whether TAT-C's access model
    (Le Moigne et al., 2017; Bardaji et al., 2024) adequately represents MSI's pointing constraints.
    Both are upstream of the dataset and out of scope.

---

## 12. Summary of defects

| # | Defect | Primary evidence | Severity |
|---|---|---|---|
| 1 | Broadband-OLR threshold cannot separate deep convection from cold surfaces | `tb < 220 K` ≡ `lwtup < 132.83 W m⁻²`; 30.5% of detections in a 5.0%-of-Earth polar band (6.11× enrichment) | Critical |
| 2 | Latitude clip ±75° is ineffective mitigation | Removes 3.41% of area for 16.62% of rows (4.9× cost ratio); ≥37.9% of Antarctica retained; polar diagnostic bins forced to 0.0% | Moderate |
| 3 | Connected-component + dissolve amplifies false positives; whole-cluster features credited to grazing passes | Frame-0 components of $9.9\times10^{5}$ km²; observed rate 1.42% vs 0.765% point-like (1.86× size bias); no intersection weighting in cell 13 | High |
| 4 | Date remapping collapses 5,880 frames to 24 atmospheric states | Notebook prints `5880 frames -> 24`; $1{,}354{,}115/245 = 5{,}527$ exactly; 11.0× unexplained hemispheric asymmetry | Critical |
| 5 | RL formulation: action-independent transitions — firing the CPR consumes no resource, so gating is a bandit, not a scheduling problem; observation carries no radiometry; label defined as `> 0` | `self._index += 1` unconditional; break-even belief $0.1/(1.101+b) \le 9.1\%$ forces a ≈100% duty cycle; observation is geometry + clock only; label entropy 0.0174 bits | High |
| 6 | Available discriminators unused | 71 variables in DDS; `lwtupclr`, `cldtmp`, `preccon`, `cape`, `cldhgh`, `tauhgh`, `precsno` all present and unread; `tautot` is fetched (cell 6) then discarded despite ROC-AUC 0.926 vs `lwtup` 0.565 | High (as opportunity) |

*Not defects (recorded to prevent re-litigation).* `spaces.Discrete(2)` is the correct action space
for CPR gating; the exclusion of `prectot`/`preccon`/`cnprcp_mean` from the observation is correct
and required; and the claim that "target prioritization is not expressible" — made in an earlier
draft of §7 — is formally retracted in §7.3.

---

## References

All entries below were verified against Crossref, arXiv, or Open Library. DOIs are given where they
exist.

Ackerman, S. A., Strabala, K. I., Menzel, W. P., Frey, R. A., Moeller, C. C., & Gumley, L. E. (1998).
Discriminating clear sky from clouds with MODIS. *Journal of Geophysical Research: Atmospheres*,
103(D24), 32141–32157. https://doi.org/10.1029/1998JD200032

Arkin, P. A. (1979). The relationship between fractional coverage of high cloud and rainfall
accumulations during GATE over the B-scale array. *Monthly Weather Review*, 107(10), 1382–1387.
https://doi.org/10.1175/1520-0493(1979)107<1382:TRBFCO>2.0.CO;2

Bardaji, J., Bayazid, A., Tapia, J. I., & Grogan, P. T. (2024). Applying the Tradespace Analysis
Tool for Constellations (TAT-C) for Earth science mission analysis. *2024 IEEE Aerospace Conference*,
1–9. https://doi.org/10.1109/AERO58975.2024.10521253

Chandak, Y., Theocharous, G., Nota, C., & Thomas, P. (2020a). Lifelong learning with a changing
action set. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), 3373–3380.
https://doi.org/10.1609/aaai.v34i04.5739

Chandak, Y., Theocharous, G., Metevier, B., & Thomas, P. (2020b). Reinforcement learning when all
actions are not always available. *Proceedings of the AAAI Conference on Artificial Intelligence*,
34(04), 3381–3388. https://doi.org/10.1609/aaai.v34i04.5740

Chun, J., Yang, W., Liu, X., Wu, G., He, L., & Xing, L. (2023). Deep reinforcement learning for the
agile Earth observation satellite scheduling problem. *Mathematics*, 11(19), 4059.
https://doi.org/10.3390/math11194059

Davis, C., Brown, B., & Bullock, R. (2006). Object-based verification of precipitation forecasts.
Part I: Methodology and application to mesoscale rain areas. *Monthly Weather Review*, 134(7),
1772–1784. https://doi.org/10.1175/MWR3145.1

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves.
*Proceedings of the 23rd International Conference on Machine Learning (ICML '06)*, 233–240.
https://doi.org/10.1145/1143844.1143874

Dessler, A. E., Yang, P., Lee, J., Solbrig, J., Zhang, Z., & Minschwaner, K. (2008). An analysis of
the dependence of clear-sky top-of-atmosphere outgoing longwave radiation on atmospheric temperature
and water vapor. *Journal of Geophysical Research: Atmospheres*, 113, D17102.
https://doi.org/10.1029/2008JD010137

Dulac-Arnold, G., Evans, R., van Hasselt, H., Sunehag, P., Lillicrap, T., Hunt, J., Mann, T., Weber,
T., Degris, T., & Coppin, B. (2015). Deep reinforcement learning in large discrete action spaces.
*arXiv:1512.07679*. https://arxiv.org/abs/1512.07679

Eddy, D., & Kochenderfer, M. (2020). Markov decision processes for multi-objective satellite task
planning. *2020 IEEE Aerospace Conference*, 1–12. https://doi.org/10.1109/AERO47225.2020.9172258

Feng, Z., Hardin, J., Barnes, H. C., Li, J., Leung, L. R., Varble, A., & Zhang, Z. (2023).
PyFLEXTRKR: a flexible feature tracking Python software for convective cloud analysis. *Geoscientific
Model Development*, 16(10), 2753–2776. https://doi.org/10.5194/gmd-16-2753-2023

Feng, Z., Prein, A. F., Kukulies, J., Fiolleau, T., Jones, W. K., Maybee, B., Moon, Z. L., Núñez
Ocasio, K. M., Dong, W., Molina, M. J., et al. (2025). Mesoscale Convective Systems Tracking Method
Intercomparison (MCSMIP): Application to DYAMOND global km-scale simulations. *Journal of Geophysical
Research: Atmospheres*, 130. https://doi.org/10.1029/2024JD042204

Fiolleau, T., & Roca, R. (2013). An algorithm for the detection and tracking of tropical mesoscale
convective systems using infrared images from geostationary satellite. *IEEE Transactions on
Geoscience and Remote Sensing*, 51(7), 4302–4315. https://doi.org/10.1109/TGRS.2012.2227762

Fiolleau, T., & Roca, R. (2024). A database of deep convective systems derived from the
intercalibrated meteorological geostationary satellite fleet and the TOOCAN algorithm (2012–2020).
*Earth System Science Data*, 16(9), 4021–4050. https://doi.org/10.5194/essd-16-4021-2024

Frey, R. A., Ackerman, S. A., Liu, Y., Strabala, K. I., Zhang, H., Key, J. R., & Wang, X. (2008).
Cloud detection with MODIS. Part I: Improvements in the MODIS cloud mask for Collection 5. *Journal
of Atmospheric and Oceanic Technology*, 25(7), 1057–1072. https://doi.org/10.1175/2008JTECHA1052.1

Herrmann, A., & Schaub, H. (2023a). Reinforcement learning for the agile Earth-observing satellite
scheduling problem. *IEEE Transactions on Aerospace and Electronic Systems*.
https://doi.org/10.1109/TAES.2023.3251307

Herrmann, A., & Schaub, H. (2023b). A comparative analysis of reinforcement learning algorithms for
Earth-observing satellite scheduling. *Frontiers in Space Technologies*, 4.
https://doi.org/10.3389/frspt.2023.1263489

Hurlbert, S. H. (1984). Pseudoreplication and the design of ecological field experiments.
*Ecological Monographs*, 54(2), 187–211. https://doi.org/10.2307/1942661

Illingworth, A. J., Barker, H. W., Beljaars, A., Ceccaldi, M., Chepfer, H., Clerbaux, N., Cole, J.,
Delanoë, J., Domenech, C., Donovan, D. P., et al. (2015). The EarthCARE satellite: The next step
forward in global measurements of clouds, aerosols, precipitation, and radiation. *Bulletin of the
American Meteorological Society*, 96(8), 1311–1332. https://doi.org/10.1175/BAMS-D-12-00227.1

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based
science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

King, J. C., & Turner, J. (1997). *Antarctic Meteorology and Climatology*. Cambridge University Press.

Le Moigne, J., Dabney, P., de Weck, O., Foreman, V., Grogan, P., Holland, M., Hughes, S., & Nag, S.
(2017). Tradespace analysis tool for designing constellations (TAT-C). *2017 IEEE International
Geoscience and Remote Sensing Symposium (IGARSS)*, 1181–1184.
https://doi.org/10.1109/IGARSS.2017.8127168

Liu, Y., Key, J. R., Frey, R. A., Ackerman, S. A., & Menzel, W. P. (2004). Nighttime polar cloud
detection with MODIS. *Remote Sensing of Environment*, 92(2), 181–194.
https://doi.org/10.1016/j.rse.2004.06.004

Machado, L. A. T., Rossow, W. B., Guedes, R. L., & Walker, A. W. (1998). Life cycle variations of
mesoscale convective systems over the Americas. *Monthly Weather Review*, 126(6), 1630–1654.
https://doi.org/10.1175/1520-0493(1998)126<1630:LCVOMC>2.0.CO;2

Mapes, B. E., & Houze, R. A. (1993). Cloud clusters and superclusters over the oceanic warm pool.
*Monthly Weather Review*, 121(5), 1398–1416.
https://doi.org/10.1175/1520-0493(1993)121<1398:CCASOT>2.0.CO;2

Nesbitt, S. W., Cifelli, R., & Rutledge, S. A. (2006). Storm morphology and rainfall characteristics
of TRMM precipitation features. *Monthly Weather Review*, 134(10), 2702–2721.
https://doi.org/10.1175/MWR3200.1

Pincus, R., Platnick, S., Ackerman, S. A., Hemler, R. S., & Hofmann, R. J. P. (2012). Reconciling
simulated and observed views of clouds: MODIS, ISCCP, and the limits of instrument simulators.
*Journal of Climate*, 25(13), 4699–4720. https://doi.org/10.1175/JCLI-D-11-00267.1

Ramanathan, V., Cess, R. D., Harrison, E. F., Minnis, P., Barkstrom, B. R., Ahmad, E., & Hartmann,
D. (1989). Cloud-radiative forcing and climate: Results from the Earth Radiation Budget Experiment.
*Science*, 243(4887), 57–63. https://doi.org/10.1126/science.243.4887.57

Reale, O., Achuthavarier, D., Fuentes, M., Putman, W. M., & Partyka, G. (2017). Tropical cyclones in
the 7-km NASA Global Nature Run for use in Observing System Simulation Experiments. *Journal of
Atmospheric and Oceanic Technology*, 34(1), 73–100. https://doi.org/10.1175/JTECH-D-16-0094.1

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S.,
Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., et al. (2017). Cross-validation strategies for
data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
https://doi.org/10.1111/ecog.02881

Rossow, W. B., & Schiffer, R. A. (1999). Advances in understanding clouds from ISCCP. *Bulletin of
the American Meteorological Society*, 80(11), 2261–2287.
https://doi.org/10.1175/1520-0477(1999)080<2261:AIUCFI>2.0.CO;2

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot
when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
https://doi.org/10.1371/journal.pone.0118432

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Turner, J., Anderson, P., Lachlan-Cope, T., Colwell, S., Phillips, T., Kirchgaessner, A., Marshall,
G. J., King, J. C., Bracegirdle, T., Vaughan, D. G., et al. (2009). Record low surface air
temperature at Vostok station, Antarctica. *Journal of Geophysical Research: Atmospheres*, 114,
D24102. https://doi.org/10.1029/2009JD012104

Ullrich, P. A., & Zarzycki, C. M. (2017). TempestExtremes: a framework for scale-insensitive
pointwise feature tracking on unstructured grids. *Geoscientific Model Development*, 10(3),
1069–1090. https://doi.org/10.5194/gmd-10-1069-2017

Wehr, T., Kubota, T., Tzeremes, G., Wallace, K., Nakatsuka, H., Ohno, Y., Koopman, R., Rusli, S.,
Kikuchi, M., Eisinger, M., et al. (2023). The EarthCARE mission – science and system overview.
*Atmospheric Measurement Techniques*, 16(15), 3581–3608. https://doi.org/10.5194/amt-16-3581-2023

Yamanouchi, T., & Charlock, T. P. (1995). Comparison of radiation budget at the TOA and surface in
the Antarctic from ERBE and ground surface measurements. *Journal of Climate*, 8(12), 3109–3120.
https://doi.org/10.1175/1520-0442(1995)008<3109:CORBAT>2.0.CO;2

Yamanouchi, T., & Charlock, T. P. (1997). Effects of clouds, ice sheet, and sea ice on the Earth
radiation budget in the Antarctic. *Journal of Geophysical Research: Atmospheres*, 102(D6),
6953–6970. https://doi.org/10.1029/96JD02866

Zhang, K., Randel, W. J., & Fu, R. (2016). Relationships between outgoing longwave radiation and
diabatic heating in reanalyses. *Climate Dynamics*, 49(7–8), 2911–2929.
https://doi.org/10.1007/s00382-016-3501-0
