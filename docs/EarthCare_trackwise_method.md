# The EarthCARE Track-Wise Method: What It Does and Why It Works

**Applies to:** the EarthCARE dataset `missions/earthcare/data/EarthCare_trackwise_decisions.parquet`,
built by `extract_trackwise.py` and `task_outputs/enrich_trackwise.py`, and analysed in
`missions/earthcare/EarthCare_trackwise.ipynb`. Baseline results are in `missions/earthcare/results/`
(produced by `python -m baselines.run_baselines --mission earthcare`). The feature tiers and
validation rules are documented in [`feature_set.md`](./feature_set.md). TRMM and GPM will get their
own datasets with the same method (`missions/trmm/`, `missions/gpm/`).

This method replaces the clustering approach described in
[`clustering_method.md`](./clustering_method.md), for the reasons given there — in short, clustering
answers "where is the storm," and this project needs "should the radar fire right now." This document
describes what the track-wise method does, and why it succeeds where clustering did not.

---

## 1. The idea in one sentence

Instead of detecting storms and then figuring out which ones the satellite observed, this method
starts from the satellite's actual flight path and asks, at every point along it: *what did the
imager see here, and what would the radar have measured here?* Every row is one such moment. No
storm has to be detected for a row to exist.

---

## 2. What it does, step by step

**1. Compute the satellite's real ground track.** EarthCARE's orbit (NORAD 59908) is propagated from
a set of 593 historical TLEs (CelesTrak, epochs 2025-05-25 to 2026-02-08), using the TLE whose epoch
is closest to each moment. The sub-satellite point (the point directly beneath the satellite) is
computed once per second across the full 8-month window, 2025-06-01 15:00 to 2026-02-01 15:00 UTC.

**2. Fix a decision every 1 minute.** The future RL agent makes a CPR on/off decision every minute
of flight. That is the row rate of the dataset: one row per 1-minute interval, **352,800 decisions**,
numbered 0 to 352,799 with **no gaps**.

**3. For each decision, fetch a small patch of weather data.** The G5NR weather simulation updates
every 30 minutes at its finest available resolution (0.0625°, roughly 7 km). For each 1-minute
decision, a small bounding box around the satellite's position at that moment is pulled from the
correct 30-minute weather snapshot — found by simple arithmetic (which 30-minute slot a given
minute falls into), not by matching timestamps, which turned out to be an important distinction
(see §5). The orbit and the weather start at the same UTC time of day and the same day of year, so
each decision reads the snapshot from the start of its own half hour (§7).

**4. Reduce that patch to a handful of numbers — the diagnostics.** The table contains GEOS-5
Nature Run diagnostics that act as MSI-like and BBR-like proxies, including cloud optical
thickness, cloud fraction, cloud-top temperature, and outgoing longwave radiation, averaged over
the 150 km MSI swath for that minute. They are not simulated EarthCARE retrievals: no instrument
forward model or retrieval chain has been applied. The future observation set is tiered; see
[`feature_set.md`](./feature_set.md). Nothing about precipitation enters the always-available
context set.

**5. Separately, record what the radar would have measured — the label.** The CPR is a
nadir-pointing instrument with a very narrow footprint, so its label is computed only from the
grid cells directly under the 60 one-second sub-satellite points of the minute, using the same
propagation as step 1. This value — precipitation, kept as `prectot_nadir` and `preccon_nadir` —
is stored but never shown to the agent as an input. A second, wider version (`_swath`) is also
stored, representing precipitation anywhere across the imager's swath, for comparison; the nadir
version is the one used to score the agent, because it is the one the CPR could have actually
measured. Using the wider version would overstate what the instrument sees, the same problem the
original clustering approach had at a much larger scale (§3.3 in the companion document).

**6. Discard the patch and move on.** Nothing is kept in memory beyond the reduced row. This is
what makes the dataset possible at all — see §3.

**7. Enrich.** After all 59 extraction blocks are saved, one script
(`task_outputs/enrich_trackwise.py`) joins them and adds land/water fractions under the nadir track,
cyclic longitude, solar hour, the next-minute longwave proxy, and validity flags (§9).

The result is one table: **352,800 rows × 40 columns**, each one a complete (diagnostic-proxy
context, radar-truth) pair at the intended decision cadence.

---

## 3. Why the data has to be fetched this way

A tempting alternative is to just download the whole weather grid for the whole period and query
it locally. At the resolution and cadence this project needs, that is not possible:

$$\underbrace{2{,}881 \times 5{,}760}_{\text{cells at 0.0625°}} \times \underbrace{13}_{\text{variables}} \times \underbrace{11{,}760}_{\text{30-min steps, 8 months}} \times \underbrace{4\ \text{B}}_{\text{float32}} \approx 10.1\ \text{TB}\ (9.2\ \text{TiB}).$$

The satellite only ever flies over a thin ribbon of the planet at any moment, so almost all of that
10 TB would be weather data the agent never needs, over parts of the globe it isn't looking at.
The fix is architectural: fetch only the small patch of grid under the satellite at each decision,
reduce it to the handful of numbers described above, and throw the rest away immediately. Peak
memory per request is under one megabyte, independent of how long the extraction runs — which is
what makes an 8-month, minute-by-minute dataset tractable on ordinary hardware, when the naive
version of the same idea is not.

---

## 4. Why this works — measured, not assumed

### 4.1 It sidesteps the clustering failure mode by construction

There is no threshold, no connected-component step, and no object anywhere in this pipeline. A row
exists because a minute of flight happened, not because a storm was detected. This removes the
entire category of problems described in the companion document — there is no mega-cluster to
merge, no whole-object area to over-credit to a grazing pass, and no question of what counts as a
"storm" to get wrong. The dataset is, by construction, exactly the shape the decision problem needs:
one row per decision.

### 4.2 The label is scoped to what the instrument can actually measure

Because the CPR's footprint is narrow, using a wide-swath label would silently teach the agent to
expect credit for precipitation the radar never flies over — the same distortion, at a smaller
scale, that made the clustering approach's whole-object crediting a problem. Restricting the label
to the true nadir track corrects this directly, and the effect is large: at every meaningful
precipitation threshold the nadir-based positive rate is well below the swath-based one, and the
gap widens as the threshold rises — nadir positives are 74% of swath positives at 0.1 mm/hr, about
half at 0.5–1 mm/hr, and only a quarter to a third at 5–10 mm/hr.

| threshold | swath positive | nadir positive |
|---|---|---|
| any precipitation | 98.6% | 97.5% |
| ≥ 0.1 mm/hr | 67.0% | 49.4% |
| ≥ 0.5 mm/hr | 41.8% | 22.4% |
| ≥ 1 mm/hr | 30.0% | **13.7%** |
| ≥ 5 mm/hr | 11.8% | 3.8% |
| ≥ 10 mm/hr | 7.6% | 2.0% |

The "any precipitation" row is not usable as a label: the model produces trace drizzle almost
everywhere, so 97.5% of nadir decisions are positive at that threshold.

At the 1 mm/hr operating threshold, **13.7% of decisions are positive** (12.8% in the Nov–Jan test
period) — a workable minority class for a reward signal, and a realistic one, since it reflects only
what the CPR could physically have observed. **54.5% of the minutes with swath rain have no rain at
nadir**, and no minute has nadir rain without swath rain (the swath contains the nadir cells).

### 4.3 The diagnostic proxies genuinely predict the label

The entire premise of CPR gating only works if cloud properties available to a passive-imager-like
context are informative about precipitation the imager cannot directly measure. This was tested on
the finished dataset with a **time-blocked split**: models are trained on June–September 2005 weather,
thresholds are set on October 2005, and everything is scored on November 2005 – January 2006, with
one weather step dropped at each boundary. Ranking the always-available features by how well they
separate precipitating from non-precipitating decisions in the test period (ROC-AUC, where 0.5 is
chance and 1.0 is perfect separation; "inverted" means lower values indicate precipitation, so the
AUC is reported as 1 − AUC):

| feature | AUC vs. nadir precipitation ≥ 1 mm/hr |
|---|---|
| cloud optical thickness (mean, `tautot_mean`) | **0.834** |
| cloud optical thickness (peak, `tautot_max`) | 0.812 |
| total cloud fraction (`cldtot_mean`) | 0.652 |
| cloud-top temperature (`cldtmp_mean`) | 0.619 (inverted) |
| nadir ocean fraction (`ocean_frac_nadir`) | 0.584 |
| outgoing longwave radiation (`lwtup_mean`) | 0.574 (inverted) |

Combining the 13 always-available inputs (the diagnostic proxies plus geometry: `lat`, `sin_lon`,
`cos_lon`, sine and cosine of `solar_hour`, and nadir land/ocean fractions):

| model | ROC-AUC [95% CI] | average precision | rainy minutes caught with the radar on 20% of the time |
|---|---|---|---|
| logistic regression | 0.837 [0.834, 0.840] | 0.409 | 60.3% |
| small neural network (64-64) | 0.889 [0.887, 0.891] | 0.527 | 65.1% |
| **LightGBM gradient boosting** | **0.893 [0.891, 0.896]** | **0.544** | **66.4%** |

against a 12.8% base rate and 19.7% for random gating. Cloud optical thickness carries the most
signal, which lines up with the physical picture: optically thick cloud is where precipitation tends
to form, so a passive imager measuring cloud thickness is measuring a real precursor of what the
radar would find. Full results are in §10.

### 4.4 The dataset spans real seasonal variation, not a single repeated snapshot

The window covers June 2005 through January 2006 continuously — **11,760 distinct 30-minute weather
states** (G5NR steps 803–12,562), not a handful of snapshots replayed many times. The starting month
was not arbitrary: all 24 months in the available G5NR record were checked with an area-weighted
global average, and June–July 2005 came out as the wettest consecutive two-month period in the
record, so the window opens on peak convective activity (the Asian monsoon and the Northern
Hemisphere ITCZ near their seasonal peak) and then sweeps forward through a full transition into
autumn and winter. That gives the dataset genuine regime diversity — different seasons, different
storm types — rather than one weather pattern repeated under many different satellite positions.
Skill holds across that variation: trained on the other months, each month scores an AUC between
0.811 and 0.844 (§10).

### 4.5 Data quality was checked and cleaned, not assumed

- **Fill values.** Some G5NR grid cells return an extreme numerical fill value in place of a true
  reading when data is missing, which is easy to miss because the value is technically a normal,
  finite number rather than an obvious error code. The extractor drops any value with magnitude
  ≥ 10⁶ before averaging, so these never enter the dataset.
- **A corrupted server response.** One request during the final run returned precipitation values of
  652–60,030 kg m⁻² s⁻¹ (the physical maximum in G5NR is about 0.07) for 3 decisions over East
  Antarctica (decisions 83775–83777, 2025-07-29 19:15–19:17 UTC orbit time). They were below the 10⁶
  cutoff, so they passed. G5NR itself was normal at that place and time (maximum 5.8 × 10⁻⁵), so the
  whole block (block 13, 6,000 decisions) was re-downloaded. The new block matched the old one in
  every value except those 6 (two precipitation columns × 3 decisions).
- **Physical-range audit.** After enrichment, every weather column is checked against its physical
  range (cloud fractions 0–1, cloud-top temperature 150–330 K, longwave 50–450 W m⁻², precipitation
  0–0.1 kg m⁻² s⁻¹, water paths 0–20 kg m⁻², optical depths 0–1,000). **No value in any of the 352,800
  rows is out of range.** The largest nadir precipitation is 247 mm/hr, a strong convective peak.
- **Targets.** All 8 target columns are finite and valid in every row; nothing is imputed.
- **Polar footprints (12 rows).** For 12 decisions at 80.8–82.2°S, where the 97° orbit turns, the
  MSI footprint polygon contained no grid-cell centre, and the extractor of the time substituted
  the cell nearest the polygon's centroid (`n_swath_cells` = 1 in those rows). The extractor no
  longer has this fallback (October 2026, with the TRMM work): an empty footprint now stops the
  run. The 12 rows are kept as extracted; re-extracting EarthCARE would first need a polar-footprint
  fix. The same change made each nadir point's grid-cell lookup wrap at ±180°, which can differ from
  the extracted rows only for nadir points within 0.03° of the dateline.
- **Missing features.** `cldtmp_mean` is missing in 17,425 rows (4.9%), where G5NR reports no
  cloud-top temperature anywhere in the swath. These are scenes without optically measurable cloud
  (median optical depth 0, none rainy); the 35% of them with cloud fraction above 0.5 are 97%
  poleward of 60°. `lwtup_fore` is missing in exactly 11,760 rows, one per weather step (§9).

---

## 5. Two things worth knowing if this pipeline is extended

**Match timesteps by position, not by date.** The satellite's ground-track clock and the weather
model's own time axis are on different calendars entirely (the orbit runs in 2025, the weather in
2005). Matching them by *finding the nearest matching timestamp* rather than by *counting how many
30-minute steps have elapsed* is a trap — an earlier version of this pipeline did exactly that, and
it silently mapped every decision in a multi-day run onto the same single weather snapshot. The fix
is to always compute the weather index by direct arithmetic (decision number → elapsed time →
weather-step number), never by nearest-timestamp lookup across two independently-referenced clocks.

**Use the true sub-satellite point for the label geometry, not the imaging footprint's shape.** At
the necessary sampling rate, consecutive one-second footprints along the track overlap and can
merge into a single shape with no way to recover the individual points that made it up. Computing
the nadir track directly from the orbit (rather than trying to infer it from the footprint
polygon's geometry) avoids this entirely and is the more robust approach regardless.

---

## 6. Summary comparison

| | clustering method | track-wise method |
|---|---|---|
| unit of the dataset | one storm object | one 1-minute decision |
| exists only if... | a threshold fires and merges cells | always — every minute of flight |
| label geometry | whole storm object, any pass touching it | true CPR nadir track only |
| positive-rate distortion | up to 39× area over-credited to a grazing pass | none — label is the actual footprint |
| temporal coverage (final run) | a single repeated weather snapshot (original notebook) | 11,760 distinct 30-minute weather states over 8 months |
| memory footprint | full field per snapshot | one small patch per decision, discarded immediately |
| demonstrated learnability | not established on the clustering table | ROC-AUC 0.893 (LightGBM), 0.837 (logistic regression) on always-available features, time-blocked test |

---

## 7. Re-extraction (October 2026)

The dataset was extracted again from scratch. This section explains why, what changed, and what it
affected. The first extraction (August 2026: 351,639 rows, 49 gaps, ROC-AUC 0.837 on a random split)
has been deleted; every number elsewhere in this document is from the new one.

### 7.1 How it was previously

Each row is built from two inputs, and each input has its own time.

**Satellite time (TATC + Skyfield).** The orbit was propagated from a single EarthCARE TLE with
epoch 2025-07-19 08:11 UTC, starting at `startdate = 2025-07-19 15:04 UTC`. Decision *i* is the
minute starting at `startdate + i min`. TATC's `compute_ground_track` produces the MSI footprint for
that minute, and Skyfield's `wgs84.subpoint` produces the per-second nadir points. The row's `time`,
`lat`, `lon`, and `solar_hour` (`UTC hour + lon / 15`) all come from this time. This part gives
EarthCARE's real local crossing times of about 14:00 (descending pass) and 02:00 (ascending pass).

TATC 3.4.10 has a setting, on by default (`repeat_cycle_for_orbit_track`), that reuses positions
from an earlier orbit if it detects a repeat cycle. Skyfield never does this. The code did not
control the setting, so the footprints and the nadir points were not guaranteed to come from the
same trajectory.

**Weather time (G5NR).** The weather snapshot is not chosen by timestamp. It is chosen by index:

```python
g5i = FIELD_START + decision // 30     # one G5NR step (30 min) per 30 decisions
ds[VARS].isel(time=g5i)
```

`FIELD_START` was **773**, which is G5NR's 2005-06-01 **00:00 UTC** snapshot.

The two times advanced at the same rate, but they started at different times of day: the satellite
at 15:04 UTC and the weather at 00:00 UTC. Nothing in the code checked that they matched.

**Downloads.** Each request fetched the weather for 3 decisions and was retried up to 4 times. If
all 4 attempts failed, the request returned an empty list, and its block of 6,000 decisions was
saved as finished anyway. On a restart, saved blocks are skipped, so the missing decisions were
never fetched again. The only protection against a hung connection was `HTTP.TIMEOUT=120` in
`~/.dodsrc`.

### 7.2 What problem it created

**Wrong time of day for the weather.** Every row paired the satellite's position with weather from
about 15 hours earlier in the day. Measured over all 351,639 rows, the offset was 904–933 minutes.
In local solar time:

| pass | satellite overhead at | weather in the row was from |
|---|---|---|
| descending | ~14:00 | ~22:40 |
| ascending | ~02:00 | ~10:35 |

This mattered for three reasons:

- Precipitation depends strongly on time of day. Convection over land peaks in the afternoon,
  which is what the 14:00 pass should see. The old rows showed late-evening weather instead, so
  every precipitation statistic was taken at the wrong hour.
- `tautot_mean` and `tautot_max` are marked daylight-only, because MSI optical depth needs sunlight.
  Rows that were daytime by orbit contained nighttime weather.
- `solar_hour` is an agent input. It matched the satellite but not the weather in the same row, so
  an agent trained on it would learn a time-of-day relationship that the real mission does not have.

The old data was not random noise. It was consistent data for a satellite crossing at ~22:40 and
~10:35. It was not EarthCARE.

**Wrong season for the orbit.** The orbit ran from July 2025 while the weather started on 1 June
2005, so the satellite's position relative to the Sun (and therefore the sunlit part of each orbit)
was about seven weeks out of step with the weather's season.

**Missing decisions.** The silently dropped requests left **49 gaps**, totalling **1,161
decisions**. Gap sizes ranged from 3 to 807 decisions. Every gap was a multiple of 3, meaning each
was one or more whole failed requests. Gaps break the 1-minute sequence that RL episodes rely on.

**Problems found while re-extracting.** The first fix (moving `FIELD_START` to 803, §7.3) was
running when a review and the run itself turned up more issues:

- **4-minute boundary error.** Step 803 is 15:00 UTC, but the orbit started at 15:04. So the
  satellite at 15:30–15:33 still read the 15:00 snapshot although 15:30 was available. 4 of every
  30 decisions (13%) used a snapshot 30 minutes older than necessary, and the offset was 4–33 min
  instead of 0–29.
- **Season mismatch after the first fix.** The intermediate fix propagated a current TLE (epoch
  2026-10-14) from 2026-10-15, so the orbit was in October while the weather was in June. The time of
  day matched, but the season did not.
- **Repeat-cycle risk.** Checked directly: for the old TLE, TATC found no repeat cycle, and TATC and
  Skyfield positions agreed to within 0.1 m at 8 dates across the window. So no data was affected,
  but a different TLE could have split the two trajectories without any error.
- **Empty responses.** The server once returned zero grid cells without an error, for 3 decisions
  (6492–6494). Fetching the same region again returned 193 × 94 cells, so it was a server fault,
  but the code treated it as a success and saved a 3-decision gap.
- **Hung downloads.** When the VPN disconnected mid-run, 12 of 18 workers were left on connections
  through an address that no longer existed. libcurl waited on them indefinitely and ignored
  `HTTP.TIMEOUT`, so a block stalled for over 90 minutes with no error and no retry.
- **Worker-pool crashes.** Resizing the parallel worker pool between steps occasionally crashed a
  worker (SIGSEGV), which made a block skip instead of retry.

### 7.3 How it was changed

Commits `a826fd8`, `4a06337`, `b816bc1`, `c39fc32`, `9dc8377`, `5e491c4`, `f5673dc`, `7d2b370`,
`8db4189`, and `76f5760`, applied to `extract_trackwise.py`:

1. **`FIELD_START` changed from 773 to 803.**
   ```python
   FIELD_START = 773   # before: 2005-06-01 00:00 UTC
   FIELD_START = 803   # after:  2005-06-01 15:00 UTC
   ```
   803 − 773 = 30 steps = 15 hours. Step 803 was checked against the G5NR server's time axis.
2. **The orbit starts on the same day of year and at the same time of day as the weather.**
   ```python
   startdate = datetime(2025, 7, 19, 15, 4, tzinfo=timezone.utc)   # before
   startdate = datetime(2025, 6, 1, 15, 0, tzinfo=timezone.utc)    # after
   ```
   Orbit 2025-06-01 15:00 UTC ↔ G5NR 2005-06-01 15:00 UTC (step 803). The calendar year differs; only
   the time of day and the season have to match. Neither 8-month span contains a 29 February, so the
   day of year stays aligned for the whole window.
3. **A historical TLE set replaces the single TLE.** `missions/earthcare/tle/sat000059908.txt` holds
   593 EarthCARE TLEs from a CelesTrak special data request (epochs 2025-05-25 to 2026-02-08, largest
   gap 2 d 13.7 h; the extractor refuses to run if any gap exceeds 3 days or the set does not cover
   the window). For each moment, TATC footprints and the per-second nadir points both use the TLE
   whose epoch is closest (TATC's `get_closest_tle_index`), so the orbit maintenance flown during the
   window is followed.
4. **TATC's repeat cycle is switched off:** `config.rc.repeat_cycle_for_orbit_track = False`. With
   it off, TATC propagates the TLEs the same way Skyfield does.
5. **A failed request now returns `None`** instead of an empty list, so a failure can be told apart
   from a request that legitimately had no rows.
6. **Failed requests are retried at block level**, for up to 6 more rounds, waiting 60 s, 2, 4, 8,
   15 and 15 min between rounds (about 45 min total).
7. **An empty response counts as a failure** and is retried.
8. **Each task has a 15-minute time limit** (`TASK_TIMEOUT = 900`). A normal task takes about 15 s.
   If any task exceeds the limit, the worker pool is killed; finished tasks are kept and the rest go
   through the retry rounds.
9. **One fixed pool of 18 workers** is used for every parallel step, so the pool is never resized.
   If the footprint step still fails, it is retried (3 attempts), and a block that cannot be saved is
   re-queued (up to 10 passes).
10. **A block is saved only if all of its requests succeeded.** Otherwise it is logged as
    `INCOMPLETE` and not saved, so rerunning the script fetches it again.
11. **New cache folder** `cache_8mo_jun2005_season`. All data from earlier attempts was deleted, and
    the extraction was restarted from the first block.

### 7.4 What the solution is

Make both times start at the same time of day (15:00 UTC) and the same day of year (1 June), follow
the real orbit with a set of TLEs, and refuse to save incomplete data.

The weather selection is still pure index arithmetic, so the §5 rule (never match timestamps across
the two time axes) still holds. With the orbit starting at 15:00 UTC and decision 0 reading the
15:00 UTC snapshot, the 30 decisions in each 30-minute block all read the snapshot from the start
of their own half hour. Both times advance together, so they stay aligned for all 352,800
decisions. Fixing the alignment through `startdate` rather than by adding an offset inside the index
formula keeps the rule "decision *i* belongs to step *i* // 30", which the block, field, and
request grouping all depend on.

For downloads, the extractor separates three cases: a request that succeeded, a request that failed
(including empty responses and tasks over the time limit), and a block that is incomplete. It keeps
retrying failures and never saves an incomplete block.

### 7.5 What it solved

**Time of day.** Every decision reads the latest snapshot at or before the satellite's time. The
offset is 0–29 minutes (mean 14.5) for all 352,800 decisions, the minimum possible with 30-minute
snapshots. Measured from the nadir track at every equator crossing over the window:

| pass | satellite overhead at | weather before the fix | weather after the fix |
|---|---|---|---|
| descending | **13:56–13:57** local | ~22:40 | 0–29 min before the pass |
| ascending | **01:56–01:57** local | ~10:35 | 0–29 min before the pass |

`solar_hour` now describes both the satellite and the weather in the same row, and daylight-only
features are now in daylight rows.

**Season.** The orbit and the weather both run from 1 June to 1 February, so the sunlit part of each
orbit matches the weather's season.

**Trajectory.** With the repeat cycle off and one TLE chosen per moment, footprints and nadir points
come from the same propagation. Footprint centroids sit 36–49 km from the mid-minute nadir point
(1st–99th percentile, median 41 km), as expected from the 5.76° MSI roll. The 5,818 decisions (1.6%)
whose centroid was more than 100 km off — near the poles and where the footprint polygon crosses
±180° — have their location replaced by the nadir point during enrichment.

**Gaps.** The dataset has **352,800 rows** (11,760 steps × 30 decisions) with **0 gaps**, instead of
351,639 rows with 49 gaps.

### 7.6 The final run

- **Duration:** 30.6 hours for all 59 blocks (2026-10-16 17:55 to 2026-10-18 00:34 MST), 18 workers.
  Blocks took 18–60 minutes (median 29), depending on load on the NCCS server.
- **Server errors:** 3,071 error pages from the server (3,064 "504 Gateway Time-out", 5 "502 Bad
  Gateway", 2 malformed), all absorbed by in-task retries. 15 blocks needed one block-level retry round; none
  needed a second. No block was incomplete or re-queued.
- **Block 13:** re-downloaded after the corrupted response described in §4.5 (57 minutes, one retry
  round). The corrupted copy is kept outside the repository for reference.

### 7.7 Impact

- **The dataset now represents EarthCARE.** The precipitation the agent is scored on, the class
  balance, and the feature-label relationships are all measured at the mission's real local times,
  in the right season, from the real orbit.
- **Earlier results were replaced.** The class-balance table, the feature AUCs, and the ROC-AUC in
  §4 were recomputed. The base rate moved from 13.0% to 13.7% and the best single feature from AUC
  0.815 to 0.834. Logistic regression gives AUC 0.837 again, but now on a time-blocked test instead
  of a random split; under the same random split the new data gives 0.833 (§10).
- **Everything downstream was regenerated**, including geometry: footprints, nadir land/water
  fractions, the location repair, `solar_hour`, `lwtup_fore`, the validity flags, the notebook, and
  the baseline results. The orbit dates in the `time` column run from 2025-06-01 15:00 to 2026-02-01
  14:59 UTC.
- **RL episodes can be continuous.** With no gaps, episodes never have to be split or padded around
  missing minutes.

---

## 8. Known limitation: diurnal coverage

Weather at a given location changes over the day. Over land, solar heating drives convection that
typically peaks in the afternoon and evening, while precipitation over the oceans often peaks near
dawn. This daily pattern is the diurnal cycle, and describing it fully requires samples across all
hours.

EarthCARE does not sample all hours. Its sun-synchronous orbit crosses each latitude at nearly the
same local solar time on every pass: about 14:00 on the descending pass and about 02:00 on the
ascending pass. In this dataset, the descending node stays at 13:56–13:57 and the ascending node at
01:56–01:57 for the whole 8-month window. Local time drifts away from these values toward high
latitudes, but repeated passes over a location keep landing in the same two local-time windows.
They do not gradually move through the day. (This concerns *which hours* are sampled, not how often
a location is revisited.) In numbers: 82% of all minutes fall in the six local hours around 14:00
and 02:00, at a mean latitude of 23° at the peak hours; the other hours are reached only near the
poles (mean latitude up to 83°). Time of day and latitude therefore cannot be separated in this
dataset.

Because the weather is matched to the satellite's local time (§7), every row describes weather in
one of those two windows only. As a result:

- **All rates in this dataset are conditional on the sampling times.** The positive rate, the
  precipitation statistics, and the feature-label relationships describe precipitation present at
  ~14:00 and ~02:00 local time. They are not daily or climatological precipitation statistics.
  Precipitation that starts and ends between the two windows, such as a late-afternoon storm over
  land, never appears in the data.
- **The gating task itself is not affected.** The radar can only measure what is beneath the
  satellite, and the satellite is only overhead at these times. The labels correctly describe what
  the CPR could have measured on each pass.
- **Results may not transfer to orbits with different local times.** A policy trained here has only
  seen ~14:00 and ~02:00 weather. A radar in a different sun-synchronous orbit, or in a
  non-sun-synchronous (precessing) orbit that samples all local times, faces a different mix of
  weather. That would need retraining or validation. TRMM and GPM, which sample all local times, are
  the planned test (`missions/trmm/`, `missions/gpm/`).

G5NR provides weather at every half hour, so this limitation can be measured in this setup: comparing
the precipitation sampled at the satellite's times with the precipitation at all hours along the
same ground track gives the fraction of daily precipitation that the two windows miss. Real
EarthCARE data could not provide that comparison.

---

## 9. Enrichment

`task_outputs/enrich_trackwise.py` builds the final table in one pass from the 59 extraction blocks
and the G5NR land/water constants (`task_outputs/fetch_g5nr_constants.py`). It checks that the
decisions are exactly 0–352,799 and that each row's weather step equals 803 + decision // 30, then:

| step | result |
|---|---|
| nadir land and water fractions | mean of the 60 one-second nadir points per decision; land = `frland` + `frlandice` (G5NR's `frland` leaves out land ice), water = `frocean` + `frlake`. Orbit-sampled means 0.339 land / 0.661 water; 17.1% of decisions are mixed (0.05 < land < 0.95). |
| location repair | 5,818 decisions whose footprint centroid is > 100 km from nadir get the mid-minute nadir point |
| cyclic longitude and solar hour | `sin_lon`, `cos_lon`; `solar_hour` = UTC hour + minute/60 + lon/15 |
| `lwtup_fore` | the next decision's swath-mean longwave, only within the same weather step: 341,040 values, 11,760 NaN (exactly one per step, the last minute before the snapshot changes) |
| validity flags | one per target; all 352,800 × 8 are valid |

It writes the parquet (SHA-256 `9fe09a93c8b0640f73f23d8422f869573ca172c90c242a4536fdd08a3eb82424`) and
`missions/earthcare/data/enrich_report.json`.

---

## 10. Baseline results

`python -m baselines.run_baselines --mission earthcare --mlp` runs experiments E1–E11 on this dataset
and writes every table and figure to `missions/earthcare/results/` with the dataset hash, package
versions and seed in `run_meta.json`. The notebook `missions/earthcare/EarthCare_trackwise.ipynb`
explains each result. The task: rain (≥ 1 mm/hr at nadir) in 12.8% of test minutes; the radar may be
on for a fraction *d* of minutes; capture@d is the share of rainy minutes it is on for.

| policy | AUC | capture@13% | capture@20% | rain amount caught @ 20% |
|---|---|---|---|---|
| random | 0.501 | 12.9% | 19.7% | 20.2% |
| geometry only (logistic regression) | 0.593 | 18.2% | 27.5% | 31.8% |
| latitude band | 0.629 | 18.5% | 28.0% | 34.5% |
| one passive feature (optical depth by day, cloud-top temperature by night) | 0.718 | 29.0% | 41.8% | 53.1% |
| logistic regression | 0.837 | 45.5% | 60.3% | 71.7% |
| small neural network | 0.889 | 50.4% | 65.1% | 71.8% |
| **LightGBM** | **0.893** | **50.8%** | **66.4%** | **72.7%** |
| oracle | 1.000 | 100% | 100% | 91.7% |

Main findings:

- **Leakage (E1).** Logistic regression scores AUC 0.833 on a random 30% split and 0.837 on the
  time-blocked split: neighbouring minutes sharing a weather snapshot do not inflate this model's
  score.
- **Seasons (E2).** Holding out each month in turn gives AUC 0.811–0.844.
- **Uncertainty (E10).** Day-block bootstrap intervals are narrow and do not overlap between
  successive policies. LightGBM beats the neural network on every resample (AUC +0.004
  [+0.003, +0.005]; capture@20% +1.4 points [+0.8, +2.0]).
- **Label geometry (E4).** Trained on nadir rain, the policy catches 66.4% of nadir rain but 51.1%
  of swath rain at 20% duty, while its precision looks like 75.0% on the swath label against 44.7%
  on nadir. For convective rain, 77% of swath-convective minutes have none at nadir and precision
  goes from 5.9% (nadir) to 25.5% (swath). The label must match the radar's footprint.
- **Night (E5).** Without optical depth (night-capable inputs), the switched day/night policy keeps
  89% of its full skill; geometry alone keeps 57%.
- **Regions (E6).** Ranking is best over land (AUC 0.950) and poleward of 60° (0.940), weakest in
  the tropics (0.846), where rain is most frequent (19.3% of minutes).
- **Features (E11).** The model depends most on `tautot_max`, `tautot_mean`, `lwtup_mean` and
  `cldtot_mean`.
- **Cost (E9).** Logistic regression has 16 parameters (3 kB), the neural network 5,249 (134 kB),
  LightGBM 18,300 (1.05 MB); all take about 1 ms per decision in Python (0.8–1.3 ms), against a
  60 s decision interval. This was timed on a laptop; timing on a Raspberry Pi–class board or a
  flight-representative processor has not been done (paper Limitation 11).

**The bar for an RL policy on EarthCARE:** beat LightGBM's capture at the same duty cycle on the same
test period (66.4% at 20%; AUC 0.893).
