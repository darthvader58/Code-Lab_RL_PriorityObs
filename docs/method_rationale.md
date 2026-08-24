# Design rationale: from threshold clustering to a fixed cell × hour table

**Scope.** This document records *why* `geos5data_0.5deg_tessellation_8mo.ipynb` was written
and how it differs from `geos5data_0.5deg.ipynb`. It is a design-rationale document: it quotes
the exact code that motivated each change, states the argument for the change, and states the
costs honestly. A separate literature-backed critique of the original method lives in
[`methodology_critique.md`](./methodology_critique.md).

> **This document is itself superseded.** The tessellation approach described here
> was replaced by a track-local design at 0.0625°/30-min resolution once the CPR-gating
> decision problem was fully specified (§1.5 below covers the framing correction that
> drove this). See [`trackwise_redesign.md`](./trackwise_redesign.md) for the current
> dataset and pipeline — this file remains as the record of why the intermediate design
> existed and what it got right and wrong.

**Companion notebooks**

| file | role |
|---|---|
| `geos5data_0.5deg.ipynb` | original pipeline — threshold + connected-component clustering |
| `geos5data_0.5deg_2yr.ipynb` | same method, but already walks the real G5NR time axis |
| `geos5data_0.5deg_tessellation_8mo.ipynb` | the redesign documented here |

**Downstream consumer.** The dataset feeds a reinforcement-learning agent that prioritises
Earth-observation targets for EarthCARE's MSI instrument (150 km swath), modelled with TAT-C.
Every design argument below is ultimately about what that agent can and cannot learn.

---

## 1. Where the problem lay

### 1.1 The detector applies a window-channel threshold to a broadband quantity

`geos5data_0.5deg.ipynb`, cell 8:

```python
threshold = 220  # brightness temperature threshold in Kelvin
...
tb = np.sqrt(np.sqrt(lwtup_2d / 5.67037e-8))   # invert Stefan–Boltzmann on broadband OLR
labels, _ = ndi.label(tb < threshold)
```

`lwtup` is **broadband** top-of-atmosphere upwelling longwave. Inverting the Stefan–Boltzmann
law on it yields a *column effective emission temperature*, not a cloud-top temperature: the
broadband flux is integrated across water-vapour and CO₂ absorption bands, so the resulting
temperature is depressed below the true emitting-surface temperature by the greenhouse effect
of the entire column.

The 220 K value, by contrast, comes from a literature calibrated on **window-channel** infrared
(~10.5–12.5 μm), where the atmosphere is nearly transparent and emissivity is close to 1 — which
is exactly why window Tb approximates cloud-top temperature and why "colder ⇒ taller cloud"
holds there. The threshold is being applied to a different physical quantity than the one it was
calibrated for.

**The consequence, arithmetically.** With σ = 5.67037 × 10⁻⁸ W m⁻² K⁻⁴:

$$T_b = \left(\frac{\text{OLR}}{\sigma}\right)^{1/4}$$

| scene | OLR (W m⁻²) | implied `tb` |
|---|---|---|
| the 220 K threshold | **132.8** | 220.0 K |
| tropical deep convection | ~90–120 | 199.6 – 214.5 K |
| clear sky, cold ice sheet (winter) | ~100–130 | 204.9 – 218.8 K |

Worked endpoints: `5.67037e-8 × 220⁴ = 132.8`; `(90 / 5.67037e-8)^0.25 = 199.6`;
`(130 / 5.67037e-8)^0.25 = 218.8`.

**Empirical corroboration.** The companion critique extracted the original notebook's own stored
latitude histogram: **30.5% of all detections fall in a polar band covering 5.0% of Earth's
surface — a 6.1× area-relative enrichment** — alongside an 11.0× hemispheric asymmetry. That is
measured from the pipeline's own output rather than argued from physics.

Deep convective anvils and clear sky over a cold, dry, high-albedo ice sheet occupy **the same
band of this variable**. No choice of threshold on `tb` alone separates them, because the
separation does not exist in the quantity being thresholded. This is a property of the variable,
not of the threshold value.

### 1.2 The latitude clip is a costly, ineffective mitigation

The response to polar false positives was to narrow the fetch:

```python
.sel(time=slice(t_min, t_max), lat=slice(-75, 75))
```

Because area on a sphere scales as sin(latitude), the retained band still covers

$$\sin(75°) = 0.96593 \Rightarrow \textbf{96.59\%} \text{ of Earth's surface}$$

while discarding 60 of 361 grid rows at 0.5° — **16.6% of the rows** to remove **3.41% of the
area**. The trade runs directly against the project's stated need for more training rows.

It also cannot work geographically: Antarctica reaches 63°S with its coastal ring at 66–70°S,
Greenland spans 60–83°N, and sea ice and winter continental interiors lie well equatorward of
75°. The cold surfaces causing the false positives are mostly *inside* the retained band.

> **Aside — a latent bug this exposed.** The clip was applied to the fetch but not to the
> precomputed geometry, which still used `slice(-89, 89)`:
>
> ```python
> sample_lat = dataset["lat"].sel(lat=slice(-89, 89)).values   # 357 rows
> area_2d = compute_grid_cell_area(sample_lat, sample_lon)
> lon2d, lat2d = np.meshgrid(sample_lon, sample_lat)
> ```
>
> The boolean cloud mask is shaped by the *fetch* (301 rows) but indexes arrays shaped by the
> *precompute* (357 rows), so `lat2d[mask]` raised
> `boolean index did not match indexed array along axis 0; size of axis is 357 but size of
> corresponding boolean axis is 301` on all 5,880 frames. Each failure was swallowed by
> `except Exception` inside `_process_frame`, which returned an empty frame, so the run only
> failed much later at `pd.concat` with the uninformative `ValueError: No objects to
> concatenate`. Fixed by aligning `sample_lat` to the fetch slice.

### 1.3 `dissolve` amplifies the false positives instead of diluting them

```python
return cells[cells.cluster > 0].dissolve(
    by=["time", "cluster"],
    aggfunc={"count": "sum", "area": "sum", "tot_prectot": "sum",
             "avg_prectot": "mean", "max_prectot": "max"},
)
```

Aggregating to **one row per connected component** inverts the usual intuition about false
positives. Per-pixel false positives would be harmless noise. Here, a uniformly sub-220 K ice
sheet collapses into a *single* polygon carrying near-zero precipitation. The largest component
visible in the original notebook's stored frame-0 output is 1,170 cells ≈ 9.9 × 10⁵ km² — an
order of magnitude smaller than the 10⁷ km² this document originally asserted, though still
10–100× the size of a large mesoscale convective system. (That figure is a *lower bound*: pandas
truncates the printed table, so the true maximum was not observed.)

Even at that reduced size the pathology holds: such a polygon almost always intersects the ground
track, so it enters the `observed == 1` class as one maximally-weighted, zero-precipitation
positive — distorting area normalisation and the reward signal far more than thousands of small
per-pixel errors would.

The same merging pathology is not confined to the poles: at a single 220 K threshold the ITCZ
routinely merges into continent-scale connected bands, so distinct storms lose individual
identity in the tropics too.

### 1.4 The time mapping collapses the dataset to 24 atmospheric states

```python
def _frame_ds_time(frame):
    # same time-mapping trick the original cell uses
    return (startdate + frame * g5nr_frame_duration).replace(
        day=20, month=5, year=2006, tzinfo=None
    )
```

`.replace(...)` overwrites day/month/year but **preserves the hour**, so every one of the 5,880
frames maps onto the 24 hours of a single day. The cell's own captured output confirms it:

```
5880 frames -> 24 distinct g5nr time slices
```

The precise claim matters here. The rows are **not** literal duplicates: the ground track differs
every hour, so the `observed` label carries genuine orbital-access information. But the
*meteorological* fields have only 24 realisations, replayed 245 times. Every `prectot`,
`tautot` and `area` value in the table is drawn from 24 spatial fields.

The consequence for the RL problem: **the effective sample size of the rare heavy-precipitation
class is capped at 24 hours regardless of how many rows the table has.** Resampling or
augmentation cannot manufacture rare-event diversity that was never fetched. Single-date
sampling additionally confounds the polar false-positive rate with season — 2006-05-20 is late
austral autumn, when the Antarctic surface is near its coldest.

This is a defect isolated to the 0.5° notebook. `geos5data_0.5deg_2yr.ipynb` already does it
correctly:

```python
def _frame_ds_time(frame):
    # walk the real g5nr hourly axis; clamp any overshoot to the final slice
    return g5nr_start + min(int(frame), n_g5nr - 1) * g5nr_frame_duration
```

The collection provides **18,288 hourly slices** spanning 2005-05-15 → 2007-06-16, so 5,880
distinct hours are available with room to spare.

### 1.5 The RL formulation — twice mischaracterised, now corrected

> **Correction (second revision).** This section has been wrong twice, in opposite directions,
> and both errors came from inferring the decision problem from the code instead of asking.
>
> 1. The first revision claimed a *fixed-action-space mismatch* — that variable-cardinality
>    clusters were incompatible with the repository's RL algorithms. The companion critique read
>    `A2C_Model.ipynb` and refuted it: `self.action_space = spaces.Discrete(2)`.
> 2. The second revision then claimed that binary action space meant *target prioritisation is
>    not expressible*, treating it as a defect. **That was also wrong.**
>
> **The actual decision problem**, as specified by the project owner: EarthCARE's MSI imager runs
> continuously and cheaply; the CPR (Cloud Profiling Radar) is the expensive, power-limited
> instrument. The agent observes MSI-measurable quantities and decides **whether to power up the
> CPR** for a given pass. The decision is genuinely binary — on or off — so `Discrete(2)` is
> *correct design, not a defect*. Likewise, `preccon`/`prectot` being absent from `_get_obs` is
> correct: they are the label, and exposing them would be leakage.
>
> §7 of [`methodology_critique.md`](./methodology_critique.md) still contains the second
> mischaracterisation and should be read with this correction in mind.

The consequences for dataset design are the opposite of what the earlier revisions implied, and
they are favourable:

- **No detector is needed to define targets.** The label is precipitation, read directly from
  `prectot`/`preccon`. The entire 220 K threshold debate (§1.1, §3.3) concerns how to *detect*
  convection — a question this formulation does not need to answer. Cloud-top temperature becomes
  a **feature**, not an arbiter of truth.
- **Feature/label separation is the load-bearing constraint.** Only MSI-observable quantities may
  enter the observation vector: `tautot`, `cldtot`, `cldhgh`, `cldtmp`, `lwtup`, `cre`, plus
  geometry and solar hour. `prectot`, `preccon` and `cape` are label-only. The notebook asserts
  this separation explicitly rather than trusting convention.
- **The tessellation is the right substrate**, for a reason unrelated to the original argument: a
  per-cell feature table aggregated per pass is exactly the (features, action, reward) tuple this
  problem needs.

#### Is the decision learnable at all?

A binary gating policy is only trainable if MSI-observable features actually separate
precipitating from non-precipitating scenes. Measured over 2,270,592 cell-hours (single-feature
AUC; values below 0.5 indicate inverse prediction):

| feature | prectot ≥ 0.1 mm/hr | prectot ≥ 1 mm/hr | preccon ≥ 0.1 mm/hr |
|---|---|---|---|
| **`tautot`** | **0.883** | **0.887** | **0.926** |
| `cre` | 0.785 | 0.866 | 0.773 |
| `cldtot` | 0.731 | 0.811 | 0.585 |
| `cldhgh` | 0.638 | 0.726 | 0.730 |
| `cldtmp` | 0.333 *(inv. 0.667)* | 0.252 *(inv. 0.748)* | 0.222 *(inv. 0.778)* |
| `lwtup` | 0.367 *(inv. 0.633)* | 0.263 *(inv. 0.737)* | 0.435 |

A logistic model on all six observable features reaches **ROC-AUC 0.935** for `prectot ≥ 1 mm/hr`
against a 1.94% base rate (average precision 0.163, an **8.4× lift** over chance), and ROC-AUC
0.870 for `preccon > 0`.

**The problem is learnable.** Cloud optical thickness carries most of the signal, which is
physically sensible — optically thick cloud is where precipitation forms. Note that `cldtmp`
and `lwtup` predict *inversely* (colder ⇒ wetter), so they are informative features even though
neither works as a standalone detector.

Two caveats. Average precision of 0.163 means precision stays modest at this base rate: a useful
CPR-gating policy will trade recall for precision, and the operating point is a duty-cycle
decision rather than a modelling one. And these figures come from a 9-day May 2005 window; they
should be re-measured across seasons before being relied on.

*Verification scope:* the `Discrete(2)` reading is established for `A2C_Model.ipynb` only. Six of
the seven RL/classifier notebooks are macOS *dataless* (cloud-evicted) in this working copy and
could not be read.

---

## 2. Why "it's the conventional method" does not settle the question

Infrared brightness-temperature thresholding **is** conventional, and the objection deserves a
direct answer rather than a dismissal. The lineage is real: the GOES Precipitation Index
(Arkin & Meisner, 1987) uses the fraction of area colder than 235 K as a rainfall proxy;
interpolated-OLR convection proxies (Liebmann & Smith, 1996) are standard in tropical
meteorology; and MCS tracking frameworks (TOOCAN, PyFLEXTRKR) are built on IR Tb thresholds,
typically 235/241 K for the cloud shield and 210–220 K for cold cores. "Threshold, label,
track" is a mature published methodology, and 220 K is a recognised cold-core value.

Conventionality, however, is evidence about a method's **preconditions**, not a property of the
method in isolation. The relevant question is whether those preconditions hold here. Three do
not:

1. **Wrong radiometric quantity.** The convention is calibrated on window-channel Tb ≈
   cloud-top temperature. This code applies it to broadband-OLR-derived effective temperature
   (§1.1). The threshold value does not transfer between the two.

2. **Wrong domain.** The OLR-proxy literature is overwhelmingly tropical (30°S–30°N, often
   15°S–15°N), and that restriction is not incidental — the OLR↔convection relationship degrades
   precisely where the surface itself is cold. Pole-to-pole application is the departure from
   convention, not the correction of it.

3. **The proxy's motivating constraint is absent.** IR thresholds exist because a radiometer in
   orbit *cannot observe precipitation or vertical velocity directly*, so the field substituted a
   radiative proxy. This pipeline reads a **model nature run** in which `preccon` (convective
   precipitation), `cape`, and `cldtmp` (cloud-top temperature) are stored fields. Using a proxy
   for a quantity that is present as ground truth in the same file inverts the epistemics the
   proxy was designed for.

**The fair counterargument to (3).** OSSE nature runs exist specifically to *simulate what an
instrument would observe*. If the research intent is to train an agent on the information a real
satellite would actually have, then deliberately restricting the pipeline to radiative proxies is
legitimate and (3) does not apply. That reading is coherent and should be stated explicitly if it
is the intent. It does not, however, rescue (1) or (2): even under an
emulate-the-instrument framing, the correct emulation of a window-channel radiometer is a
window-channel-like variable such as `cldtmp`, not broadband OLR, and the domain restriction
still applies.

---

## 3. How the new method works

### 3.1 The frame index *is* the time index

```python
g5nr_times = pd.to_datetime(dataset["time"].values)
assert HOURS <= len(g5nr_times), f"only {len(g5nr_times)} hourly slices available"
```

Walking the real axis from index 0 means frame `f` **is** G5NR time index `f`. The
`_frame_ds_time` helper and the whole date-remapping concept disappear. Distinct atmospheric
states go from **24 → 5,880**.

### 3.2 No object detection: `to_dataframe()` is the tabularisation

```python
sub = fetch(t0, min(t0 + CHUNK_HOURS, HOURS))
df = sub.to_dataframe().reset_index()[["time", "lat", "lon"] + VARS]
df[VARS] = df[VARS].astype("float32")
df.to_parquet(path, index=False)
```

One row per `(time, lat, lon)`. Roughly six lines replace ~150 lines of masking, labelling,
geometry construction and dissolving. There is no threshold, so there is no threshold to be
wrong.

### 3.3 The detector: the same 220 K threshold, applied to cloud-top temperature

The redesign does **not** abandon thresholding. It keeps the conventional 220 K value and
changes the variable it is applied to:

```python
cells_df["convective"] = (cells_df["cldtmp"] < 220).astype(np.int8)
```

`cldtmp` is confirmed by the collection's `.das` as:

```
cldtmp {
    String units "K";
    Float32 _FillValue 1.0E15;
    Float32 missing_value 1.0E15;
    String long_name "cloud_top_temperature ";
}
```

This is the quantity the 220–235 K literature is calibrated on, so this detector is *more*
conventional than the broadband inversion in §1.1, not less. And it is structurally immune to
the polar false positive: a cloud-top temperature is only defined where there is a cloud top,
so bare cold ice has nothing to report.

**The threshold is safe on both fill paths.** Where there is no cloud top the value is
`1.0E15`. If xarray applies `mask_and_scale` (the default) it becomes `NaN`, and `NaN < 220`
is `False`. If the fill ever survives unmasked, `1e15 < 220` is also `False`. Either way clear
sky is excluded, with no explicit masking code required. This was verified by running the
detector against both representations and confirming identical output.

**The assumption was checked — and it FAILED.** The `.das` declares a fill value but does not
prove it is used for clear sky. Run against real G5NR data, the notebook's verification cell
reported:

```
cldtmp valid overall      :  86.8%
  where cldtot < 0.01     :  18.3%   <- expect ~0
assumption holds: False
```

`cldtmp` carries a value under clear sky in 18.3% of clear cells, so it is **not** a pure cloud
mask in this collection. The detector is therefore gated on cloud fraction:

```python
CLDTOT_GATE = 0.5
cells_df["convective"] = ((cells_df["cldtmp"] < 220)
                          & (cells_df["cldtot"] > CLDTOT_GATE)).astype(np.int8)
```

**The gate turned out to be nearly a no-op**, which is itself the finding: it removes only 5.8%
of detections (277,460 → 261,447) and just 0.7% of polar ones (14,778 → 14,671). Cloud fraction
over the Antarctic in May is genuinely high, so `cldtot > 0.5` does not discriminate there.

The verification code that produced this:

```python
valid = cells_df["cldtmp"].notna() & (cells_df["cldtmp"] < 1e14)
clear = cells_df["cldtot"] < 0.01
ASSUMPTION_HOLDS = valid[clear].mean() < 0.05
```

If the assumption fails, the cell prints an explicit warning directing the reader to gate the
detector with cloud fraction — `(cldtmp < 220) & (cldtot > 0.5)` — rather than failing silently.

### 3.4 The old threshold is relocated, not removed

This is the most-misread part of the design, so it is stated explicitly. Both of these are
computed as **derived columns**:

```python
# cloud longwave radiative effect: ~0 for clear sky (including cold ice sheets),
# 60-100 W/m2 under deep convection.
cells_df["cre"] = cells_df["lwtupclr"] - cells_df["lwtup"]

# OLD: what the previous notebook called a detection. Kept so its label set stays
# exactly reproducible from this table (cells_df.old_hit == 1).
cells_df["tb_old"] = (cells_df["lwtup"] / 5.67037e-8) ** 0.25
cells_df["old_hit"] = (cells_df["tb_old"] < 220).astype(np.int8)
```

Therefore `cells_df[cells_df.old_hit == 1]` **reproduces the original label set exactly**. The
reverse is impossible: the original table contains no rows warmer than 220 K, so that information
is destroyed at construction time and cannot be recovered without refetching.

| | original | redesign |
|---|---|---|
| when the 220 K decision is made | at construction, irreversibly | as a derived column, any time |
| data above the threshold | destroyed before the agent sees it | retained |
| can it reproduce the other dataset? | no | **yes** |

The new dataset is a strict information superset. That is what makes threshold-sensitivity
ablations (210 / 220 / 235 K), detector comparisons, and imbalance tuning possible without
re-reading 1.7 GB from a server that has already proven unreliable.

### 3.5 `observed` via spatial join on a static grid

The original label required a per-hour geometric union and a vectorised intersects:

```python
gt_per_hbin = ground_tracks.assign(hbin=gt_hbin).dissolve(by="hbin")[["geometry"]]
merged = clusters.assign(hbin=clusters_hbin).merge(
    gt_per_hbin, left_on="hbin", right_index=True, how="left", suffixes=("", "_gt"))
observed[has_gt] = shapely.intersects(left_geom[has_gt], right_geom[has_gt]).astype(int)
```

> **Correction.** This section originally claimed that a fixed grid removes the need for geometry
> operations, replacing them with integer binning. **That was wrong, and it produced a 7.2×
> undercount.** `compute_ground_track` returns one row per *10-minute pass*, stored as a
> MULTIPOLYGON spanning ~4,000 km of orbit — not a small per-timestep footprint. The bounding box
> of such a geometry is enormous and its corners fall nowhere near the actual swath, so
> corner-based binning marked ~4 cells per pass instead of ~29. Measured effect: 22.9 observed
> cells per frame instead of 289.6. The label now uses a real spatial join against a static cell
> grid, which keeps the fixed-grid benefit (the grid is built once and reused) without the false
> assumption:

The static grid still helps — it is constructed once — but the intersection itself must be
geometric:

```python
cell_boxes = gpd.GeoDataFrame(                       # built once, reused every frame
    {"ci": np.repeat(np.arange(n_lat), n_lon),
     "cj": np.tile(np.arange(n_lon), n_lat)},
    geometry=[box(x, y, x + CELL_DEG, y + CELL_DEG) for y in lat_c for x in lon_c],
    crs="EPSG:4326")

pairs = gpd.sjoin(cell_boxes, gt_sub, how="inner", predicate="intersects")
seen  = np.unique(key(pairs["frame"].values, pairs["ci"].values, pairs["cj"].values))
cells_df["observed"] = np.isin(key(cells_df["frame"].values, ci_, cj_), seen).astype(np.int8)
```

This is exact rather than approximate, and it handles the antimeridian correctly through real
geometry instead of a modulo trick. It is also fast: the join over 1,296 passes and 10,512 cell
boxes completes in ~2 s using geopandas' spatial index.

### 3.6 Fetching within NCCS rate limits

```python
def fetch(t0, t1):
    """One bounded, strided slab. Retries with backoff, reopening the DAP handle."""
    global dataset
    for attempt in range(5):
        try:
            return (dataset[VARS]
                    .isel(time=slice(t0, t1),
                          lat=slice(None, None, STRIDE),
                          lon=slice(None, None, STRIDE))
                    .load())
        except Exception as e:
            print(f"  retry {attempt + 1}/5: {type(e).__name__}: {e}", flush=True)
            _time.sleep(min(5 * 2 ** attempt, 60))
            dataset = open_g5nr()   # a stalled connection poisons the handle
```

Four properties keep this inside the server's tolerance, three of them adopted directly from the
hardening already present in `geos5data_0.5deg_2yr.ipynb`:

- **strided** — requests the 2.5° subsample rather than the full 0.5° field
- **small chunks** — `CHUNK_HOURS = 72`, with `PAUSE = 2.0 s` between requests
- **checkpointed** — one parquet file per chunk, so a rerun resumes rather than refetching
- **`HTTP.TIMEOUT` in `~/.dodsrc`** — converts libnetcdf's indefinite hang on a stalled transfer
  into a catchable exception

Transfer budget, 8 variables at `STRIDE = 5`:

```
8 vars × 10,512 cells × 4 bytes         =  328 KiB / hour
                              × 72      = 24.2 MB / request
                              × 5,880   = 1.98 GB total
```

versus ~48.9 GB at full 0.5° resolution — a 25× reduction, and the reason an 8-month pull is
attemptable at all.

---

## 4. Side-by-side

| property | original | redesign |
|---|---|---|
| detector | `220 K` on **broadband-derived** tb | `220 K` on **cloud-top** `cldtmp` |
| polar false positives | clear ice indistinguishable from anvils | `cldtmp` undefined without a cloud top |
| distinct atmospheric states | **24** | **5,880** |
| rows | ≈1.45 M clusters¹ | **61.8 M** cell-hours |
| target identity across time | none | stable cell id |
| action-space cardinality | varies per hour | fixed |
| `observed` cost | per-hour union + intersects | integer binning |
| class imbalance | fixed by 220 K | percentile dial (`Q = 0.99`) |
| threshold reversible? | no | yes |
| transfer for 8 months | ~48.9 GB @ full res | ~1.98 GB |

¹ Not independently measured — taken from the comment in cell 10 of the original notebook
(`"pays the unary_union cost ~5,880 times instead of ~1.45M times"`), which implies ~1.45 M
cluster rows in an earlier successful run.

---

## 5. What the redesign gives up

Stated plainly, because these are real losses and not rhetorical concessions:

1. **Object identity and lifecycle.** Connected components yield storms that can be tracked
   across time with area, growth rate and age. A cell table has none of that. If the science
   framing is "observe *storms*", this discards a mature body of MCS-tracking methodology.
2. **Meteorological interpretability.** "This polygon is a convective system" is explainable to a
   reviewer; "cell 4,412 at hour 3,001" is not.
3. **A new imbalance.** 61.8 M rows of predominantly clear sky is its own pathology — it is
   controllable via the percentile dial, but it does not vanish.
4. **Subgrid extremes.** `STRIDE = 5` point-samples every fifth native cell rather than taking a
   box mean or max. Precipitation is spatially intermittent, so peak intensity within a 2.5° cell
   is systematically under-represented. This is an unresolved cost, accepted to make the transfer
   budget feasible.

---

## 6. Adopted approach: keep the threshold, fix its input

**Status: implemented, tested, and partially disconfirmed — see §6.1.** The primary detector is `cldtmp < 220` (§3.3). The choice is not
"no threshold" — it is the same conventional threshold applied to the variable the literature
is actually calibrated on, which makes it *more* conventional than the broadband inversion it
replaces while being structurally free of the polar false positive.

The `.das` has since been retrieved and confirms `cldtmp` as `cloud_top_temperature` in K with
`_FillValue = missing_value = 1.0E15`. What metadata cannot establish is whether that fill is
actually used under clear sky, so the notebook verifies it empirically against `cldtot` and
warns loudly if the assumption fails (§3.3).

### 6.1 What the measured run showed

Executed over 216 frames (2005-05-15 → 05-24), 2,270,592 cell-hours:

| latitude band | old flag % | old dry % | new flag % | new dry % |
|---|---|---|---|---|
| < −75 | 53.91 | 100.00 | **7.63** | 99.99 |
| −75:−45 | 3.85 | 97.43 | 15.36 | 95.54 |
| −45:−15 | 0.61 | 50.71 | 9.24 | 59.10 |
| tropics | 1.06 | 3.48 | **20.64** | **11.71** |
| 15:45 | 0.42 | 45.11 | 12.70 | 63.79 |
| 45:75 | 0.48 | 82.95 | 7.05 | 85.57 |
| > 75 | 0.46 | 96.88 | **1.18** | 100.00 |

**What worked.** Polar flagging fell from 53.91% to 7.63% of cells below 75°S — an 86% reduction
in absolute detections (117,378 → 16,613). In the tropics the new detector flags 20.64% of cells
against the old detector's 1.06%, and only 11.71% of those lack convective precipitation. As a
tropical convection detector it is decisively better.

**What did not work.** `new_dry_%` remains ~100% at both poles, and the aggregate poleward of 45°
is unchanged at 94.1%. The `cldtot` gate removed only 5.8% of detections overall and 0.7% of
polar ones.

**The honest interpretation.** `cldtmp < 220` detects cold **cloud**, not **convection**. High
latitudes genuinely have cold cloud tops — frontal systems, cirrus — but their precipitation is
stratiform snow, not convective. A near-100% `dry` rate there is therefore substantially *correct
physics* rather than pure detector error: the metric asks "is this convective?" of a detector that
answers "is this a cold cloud?". The residual polar detections are far fewer and better justified
than the original's, but they are still not convection.

**Consequence for the recommendation.** If the target is specifically convection, cloud-top
temperature is necessary but not sufficient; a convective gate (`cape`, or `preccon` itself when
it is not the label) is required. If the target is *observable cold cloud*, the gated `cldtmp`
detector is appropriate as it stands. This choice should be made explicitly rather than inherited.

The end state still treats the cell table as **substrate rather than replacement**: retain
raw fields per cell, and derive objects on top by thresholding `cldtmp` and running the same
`ndi.label` whenever storm objects are wanted. Object definition becomes a hyperparameter instead
of a permanent property of the data, and both framings remain available from one dataset.

Other discriminators available in the same collection and currently unused: `preccon`
(convective precipitation), `precsno` (snow vs. rain), `cape`, `cldhgh`/`cldmid`/`cldlow`,
`tauhgh`.

---

## 7. How to falsify this

None of the above should be settled by argument. The diagnostic cell exists to decide it from
the data:

```python
old_hit = cells_df.tb_old < 220          # old detector
dry     = cells_df.preccon <= 1e-8       # no convective precip at all
```

It reports, per latitude band, the fraction of cells the **old** detector flagged that carry
**zero convective precipitation** — i.e. its false-positive rate.

- If `of_those_dry_%` is low at all latitudes, the central claim of this document is **wrong** and
  the original method is sound.
- If it rises sharply poleward of ~45°, the claim is established from the project's own data
  rather than from the reasoning above.

**This has not yet been run against real G5NR data.** Until it is, §1.1 remains a physical
argument supported by arithmetic, not an empirical result.

---

## 8. Verification status

Being explicit about what is measured versus argued:

| claim | status |
|---|---|
| `tb` arithmetic (220 K ⇒ 132.8 W m⁻²; 110 W m⁻² ⇒ 209.9 K) | **verified** numerically |
| grid: 73 × 144 = 10,512 cells ⇒ 61.8 M rows at 5,880 h | **verified** |
| cell-index round-trip (no off-by-one) | **verified** on synthetic data |
| `observed` binning, diagnostic table, parquet IO | **verified** on synthetic data |
| `5880 frames -> 24 distinct g5nr time slices` | **verified** — original notebook's own output |
| shape mismatch 357 vs 301 | **verified** — original traceback |
| OLR ranges for convection and cold ice sheets | literature-typical values, **not** measured here |
| OpenDAP honours the stride on the wire | **unverified** — watch smoke-test chunk timings |
| `cldtmp` declared `_FillValue`/`missing_value` = 1.0E15 | **verified** — collection `.das` |
| `cldtmp` fill actually used under clear sky | **REFUTED** — valid over 18.3% of clear cells |
| `cldtot > 0.5` gate removes polar false positives | **REFUTED** — removes 0.7% of polar flags |
| cldtmp detector cuts polar flagging vs. broadband | **verified** — 53.91% → 7.63% of cells |
| tropical detection quality | **verified** — 20.64% flagged, 11.71% dry (old: 1.06% / 3.48%) |
| bbox-corner `observed` binning | **REFUTED** — 7.2× undercount; replaced with `sjoin` |
| corrected `observed` coverage | **verified** — 289.6 cells/frame vs 22.9 before |
| detector safe on both NaN and raw-1e15 fill paths | **verified** — identical output on both |
| ~~fixed-action-space mismatch~~ | **refuted** — `spaces.Discrete(2)`; see §1.5 |
| prioritisation not expressible in the A2C environment | **verified** — `A2C_Model.ipynb` |
| polar detection enrichment 6.1× (30.5% of hits in 5.0% of area) | **verified** — notebook's own histogram |
| largest frame-0 component ≈ 9.9 × 10⁵ km² | **verified** as a lower bound |
| old-detector false-positive rate | **not yet measured** — §7 |

## 9. Known open items

- **`cldtmp < 220` is a cold-cloud detector, not a convection detector** (§6.1). Decide explicitly
  which target the project wants, and add a `cape`/`preccon` gate if it is convection.
- The residual ~15,000 polar detections carry no convective precipitation. They are an 86%
  reduction on the original but are not zero.
- `STRIDE = 5` point-samples every fifth native cell, so subgrid precipitation extremes are
  under-represented. Unresolved.
- `SMOKE_TEST = True` is still the default: 216 of 5,880 frames (~9 days of May 2005). Every
  number in §6.1 is from that window and may not generalise across seasons.
- `ndi.label` does not wrap at the antimeridian, kept deliberately for parity with the original.

**Resolved:** the `pd.cut` left-open bug (`include_lowest=True` applied); the bbox-corner
`observed` undercount (replaced with `sjoin`).
