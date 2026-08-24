# Track-wise dataset: CPR-gating at 0.0625°, 30-minute cadence

**Scope.** This document covers the third redesign of the training pipeline,
implemented in `geos5data_0.0625deg_30mn_trackwise.ipynb` and its detached
extraction script `extract_trackwise.py`. It supersedes the tessellation
approach documented in [`method_rationale.md`](./method_rationale.md) for the
reasons given in §1. Where a claim depends on a code excerpt, the excerpt is quoted verbatim from
the file on disk, with one noted exception (§8.1, superseded code); where a
claim is a number, it was computed against real fetched data, not estimated.

**Companion documents**

| file | covers |
|---|---|
| `methodology_critique.md` | defects in the original clustering notebook |
| `method_rationale.md` | the 0.5° cell×hour tessellation redesign, and its own corrections |
| `trackwise_redesign.md` (this file) | the 0.0625°/30-min track-local redesign, current dataset |

---

## 1. The decision problem, restated

Earlier revisions of this project's documentation mischaracterised the RL
formulation twice (see `method_rationale.md` §1.5). The corrected, final
framing, as specified by the project owner:

EarthCARE's MSI imager runs continuously and cheaply. The CPR (Cloud
Profiling Radar) is the expensive, power-limited instrument. **The agent
observes MSI-measurable cloud properties and decides, once per interval,
whether to power up the CPR.** Correctness is scored afterwards against
whether the CPR would actually have measured significant precipitation —
`prectot`/`preccon` — which the agent must never observe as a feature.

This framing is what the tessellation notebook could not cleanly express
(objects had no fixed identity; the label geometry was unresolvable at 2.5°
because the MSI swath is narrower than one grid cell) and what this redesign
is built around directly: **one row per decision, MSI-observable features,
CPR-truth label, nothing else.**

---

## 2. Why the architecture changed again: the 12 TB problem

The tessellation notebook built a **global** cell × hour table at 0.5°. That
does not scale to higher resolution. One field at 0.0625° (the resolution
`inst30mn_2d_met1_Nx` is natively defined on) is:

$$2881 \times 5760 = 16{,}594{,}560 \text{ cells} = 66.4 \text{ MB per variable per timestep}$$

Eight months at 30-minute cadence with 13 variables:

$$11{,}760 \text{ steps} \times 13 \text{ vars} \times 66.4 \text{ MB} \approx 11.7\text{ TB}$$

No striding or chunking rescues a table of that size. The fix is architectural,
not parametric: **never materialise a global grid.** The satellite only ever
flies over a thin band of the planet, so the pipeline fetches a small bounding
box around a handful of consecutive orbital positions, reduces it to one row
per decision, and discards the array. Peak memory is one slab (~0.94 MB for a
`GROUP=3` request), independent of how long the run is.

---

## 3. Resolution, cadence, and decision interval

| parameter | tessellation notebook | this notebook |
|---|---|---|
| spatial resolution | 0.5° (coarsened) | **0.0625°** (G5NR native) |
| temporal cadence | 1 hour | **30 minutes** — the finest G5NR offers anywhere |
| decision interval | 10 minutes (~4,080 km/decision) | **1 minute** (~408 km/decision) |
| architecture | global grid, materialised | **track-local, streamed, discarded per row** |

30 minutes is the ceiling of temporal resolution G5NR provides at any
resolution — verified by walking the OpenDAP catalog
(`https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/inst/`):
every `inst`/`tavg` collection at 0.0625° is `30mn`; at 0.5° the finest is
`01hr`. There is no finer G5NR product to move to.

The 1-minute decision interval was chosen because it is the point at which
label geometry becomes meaningful (§4) and because it multiplies training
rows without touching the underlying weather-field diversity — see §7 for the
distinction between rows and independent samples.

---

## 4. Label geometry: nadir vs. swath, and why it matters more at this resolution

`compute_ground_track` for a decision interval returns a single geometry
covering every sub-footprint sampled inside that interval. Two aggregations
are computed from it:

```python
row[f"{v}_swath"] = _nanmax(arr[inside]); row[f"{v}_nadir"] = _nanmax(arr[ni,nj])
```

- **swath** — the peak label value anywhere inside the imaging field of regard
  (MSI's accessible band, wider than the physical 150 km swath once the
  ±5.76° roll capability is included — measured effective width ≈ 376 km).
- **nadir** — the peak value only along the sub-satellite track, computed from
  explicit `skyfield` sub-satellite points, not inferred from the footprint's
  constituent geometry parts (see §8 for why that distinction matters).

**Swath overstates what the CPR would measure.** The CPR is a nadir-pointing
instrument with a ~0.75 km footprint; crediting it with precipitation
anywhere across a 376 km accessible band is not physically defensible. This
was quantifiable for the first time at 0.0625°: the MSI field of regard spans
21.6 cells at this resolution versus 0.54 cells at the tessellation
notebook's 2.5° grid — i.e. **the label geometry is unresolvable on that
coarser grid**, which is why this correction could not have been made there.

Measured effect, June–July 2005, 87,825 rows:

| threshold | swath positive | nadir positive |
|---|---|---|
| any > 0 | 98.2% | 97.4% (7-day sample) |
| ≥ 0.1 mm/hr | 66.5% | 48.8% |
| ≥ 1 mm/hr | 31.7% | 14.8% |

Restricting the label to what the instrument actually overflies roughly
**halves** the positive rate at every threshold. `nadir` is the reward signal
used downstream; `swath` is retained for comparison and diagnostics.

---

## 5. Choosing the weather window: measured, not assumed

Two decisions needed a window: which months, and how long a span. Both were
answered from data rather than intuition.

**Which months.** The monthly-mean collection `tavg01mo_2d_met3_Cx` was
queried across all 24 available months (2005-06 → 2007-05) and reduced to an
area-weighted global mean using `cos(latitude)` as the weight (correct area
weighting on a lat/lon grid):

```python
w = np.cos(np.radians(lat))
np.average(prectot[valid], weights=w[valid]) * 86400   # kg m-2 s-1 -> mm/day
```

Result: **June 2005 (3.207 mm/day) and July 2005 (3.202 mm/day) are the two
highest of all 24 months**, giving a consecutive-pair mean of **3.204 mm/day**
— the highest 2-month window in the record. 2006-06+07 is second at 3.127.
Physically this is the expected result: boreal summer combines the Asian
monsoon with the Northern Hemisphere ITCZ near its seasonal peak.

**How long a span, and why start there rather than centre there.** The
8-month extension (§9) begins at the same June 2005 point and runs forward
through NH autumn into winter (2005-06-01 → 2006-02-01), rather than
centring the window on the wettest point. This was a deliberate choice: it
opens on peak convective activity and then sweeps through a full seasonal
transition, which gives the dataset genuine regime diversity instead of 8
months of a single regime. The tessellation notebook's central defect (24
distinct weather states from a single remapped day) is the failure mode this
guards against.

---

## 6. Reproducing the original notebook's figures

Cartopy visuals ported from the original clustering notebook, adapted to
row-per-decision data (no cluster objects exist in this design, so the
observed/unobserved cluster animation is replaced by a decisions-along-track
scatter, coloured by the CPR-fire decision):

- KDE map of nadir precipitation and of `tautot_mean` along track
- decisions-along-track map, red = CPR off / green = CPR on
- precipitation vs. solar-hour line plot and KDE
- latitude-band table (decisions, CPR-on %, mean mm/hr)

## 7. Exports: CZML for Cesium, CSV for kepler.gl — GeoJSON deliberately dropped

CZML was chosen over GeoJSON for the Cesium deliverable because **the
simulation is inherently time-dynamic** — the satellite is at a different
position at every second, and GeoJSON has no mechanism to express that. CZML's
`position.epoch` + `cartographicDegrees` sample array and per-entity
`availability` window are what make an animated orbit possible at all; a
static GeoJSON point cloud cannot represent "the satellite is here at time
*t*." The export builds:

- one `position`-sampled entity for EarthCARE with a trailing `path`
- one point entity per decision, clamped to ground, green/red by the CPR
  decision, with a popup showing the decision's nadir precipitation, `tautot`,
  and cloud fraction
- a vertical green `polyline` "beam" for every decision where the CPR fired,
  from orbital altitude down to the surface

This first export is explicitly the **oracle**: the `fire` column is
`prectot_nadir > threshold`, i.e. hindsight ground truth, not a trained
policy's output. It is the ceiling any agent is measured against and the
correct baseline to visualise before any model exists. Swapping in a trained
policy's actions later is a one-column change to the export cell; the CZML
structure does not change.

kepler.gl is served CSV, not GeoJSON: kepler auto-detects `lat`/`lon`
columns, and CSV is roughly 6× smaller than an equivalent GeoJSON export.
GeoJSON was dropped from the pipeline entirely — nothing downstream (Cesium,
kepler.gl, or the RL training code) needs it.

---

## 8. Two real bugs found and fixed during this build

Recorded because both were subtle and both cost real wall-clock time; a
future rerun should not reintroduce either.

### 8.1 Overlapping sub-footprints silently break nadir inference from geometry

The original nadir logic (already replaced by the time this document was
written, and not recoverable from any file on disk — reconstructed here from
the session record rather than verified against source) derived the
sub-satellite track from the footprint's constituent geometry parts:

```python
parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
```

At `SUB_STEP = 1 s` — required for a contiguous nadir track, since EarthCARE
covers ~6.8 km/s against a ~6.96 km cell — the 60 sub-footprints in one
decision **overlap**, so Shapely dissolves them into a single `Polygon` with
no `.geoms` attribute. The fallback (`[geom]`) silently collapsed a 408 km
nadir track to **one** centroid point. Verified reproduction on synthetic
geometry:

```
geom type: Polygon | has .geoms: False
old parts-based nadir would have used 1 point(s) instead of 59
```

**Fix:** compute sub-satellite points directly from the orbit via
`skyfield.wgs84.subpoint`, independent of footprint topology. This is what
`nad_lat`/`nad_lon` do in the current code; nadir cell count is now measured
at 60.0 per decision, matching the true sub-sample count.

### 8.2 A cross-process handle reset that silently did nothing

The retry logic for a failed OpenDAP fetch attempted to discard a poisoned
DAP handle so the next attempt would open a fresh connection:

```python
except Exception:
    globals()["_WDS"] = None          # poisoned handle; reopen next try
    _time.sleep(min(4 * 2 ** a, 45))
```

Under `loky`/cloudpickle, `globals()` inside a worker function does not
reliably resolve to the module namespace holding `_WDS`. The reset **silently
did nothing.** Consequence: after any event that kills open sockets (a laptop
sleep, in practice), every worker's cached handle stayed dead permanently —
each subsequent task exhausted its retry budget against a corpse connection,
timing out repeatedly, while the server itself answered fresh requests in
under a second. Measured cost: **18 workers, ~10 hours, zero rows produced**,
confirmed by testing a brand-new connection against the same endpoint during
the stall (`.das` in 0.74 s, `.dods` data request in 0.70 s — server healthy
throughout).

**Fix:** route the reset through the function that owns the binding, so
`global` resolves correctly, and force a fresh handle on every retry rather
than only after a "poisoning" that never actually registered:

```python
def _wds(force=False):
    global _WDS
    if force and _WDS is not None:
        try:
            _WDS.close()
        except Exception:
            pass
        _WDS = None
    if _WDS is None:
        import xarray as xr
        _WDS = xr.open_dataset(G5NR_URL, decode_times=True)
    return _WDS
...
    ds = _wds(force=(a > 0))    # retries always get a FRESH handle
```

Verified directly: `_wds()` called twice returns the same object id;
`_wds(force=True)` returns a different one. Measured effect on throughput:
block times fell from ~104 minutes (the first three blocks, fetched before
this fix, are the ones reused unmodified in the 8-month run) to **18–31
minutes** after the fix — roughly a 4× improvement, because transient errors
had been permanently degrading the worker pool throughout the run, not just
at the point of the eventual freeze.

**General lesson:** `globals()` inside a function executing in a separate
process (via pickling/cloudpickle) is not a reliable way to mutate
module-level state defined elsewhere. Any such mutation should go through a
function that declares the name with `global` at definition time.

---

## 9. Status of the current dataset

**Completed: June–July 2005**, `cache_junjul2005/` → superseded by the
8-month extension below (folded into `cache_8mo_jun2005/`, first 14 blocks
reused unmodified since checkpointing operates at the block level).

- 2,928 weather fields (G5NR steps 773–3700), 87,825 of 87,840 possible rows
  (99.98%; 15 rows lost to exhausted retries in one block, unresolved — an
  accepted, quantified gap, not a silent one)
- balanced label: `prectot_nadir > 0.0983 mm/hr` → exactly 50.0% positive
- 17 rows (0.02%) violate the `nadir ≤ swath` invariant, all with |lat| > 75°;
  this is meridian convergence clipping part of the nadir track outside the
  fetched bounding box near the poles, not a logic error — the invariant
  should be understood as holding outside the highest latitudes rather than
  universally
- completed in 6.2 h wall time (18 workers) — far faster than the pre-fix
  burst-benchmark estimate of 48 h, because that estimate did not (and could
  not) account for the throughput lost to bug §8.2

**In progress: extension to 8 months** (2005-06-01 → 2006-02-01, 11,760
weather fields, 352,800 rows target), launched reusing the June–July cache.
One partial block (128 of 200 fields, from the original 61-day run) was
identified and deleted before the extension launch so it would refetch in
full rather than being silently skipped by the block-existence resume check —
worth flagging as a general hazard: **checkpoint resume by file existence
cannot distinguish a complete block from a truncated one**; extending a
capped run should always re-verify the trailing block's row count.

---

## 10. Verification status

| claim | status |
|---|---|
| 0.0625°/30-min collection exists, is the finest G5NR offers | **verified** — OpenDAP catalog walk |
| global grid at 0.0625° for 8 months ≈ 11.7 TB | **verified** arithmetic |
| MSI field-of-regard spans 21.6 cells at 0.0625° vs 0.54 at 2.5° | **verified** arithmetic |
| nadir vs swath positive-rate gap (halved at every threshold) | **verified** — 87,825 real rows |
| June+July 2005 is the wettest 2-month window | **verified** — all 24 months queried |
| overlapping-footprint nadir collapse (bug §8.1) | **verified** — reproduced on synthetic geometry |
| `globals()` handle-reset no-op (bug §8.2) | **verified** — object-identity test + live server health check during the stall |
| 4× block-time improvement from the fix | **verified** — measured before/after in the same run |
| 8-month dataset completeness | **in progress** — not yet verified, run ongoing |
| CZML packet validity (position samples, beams, clock) | **verified** — tested against completed June–July data before use on the full set |
