# The Track-Wise Method: What It Does and Why It Works

**Applies to:** `geos5data_0.0625deg_30mn_trackwise.ipynb` and its extraction script
`extract_trackwise.py`. This is the dataset-preparation output used to support later training. The
enriched feature tiers and validation rules are documented in [`feature_set.md`](./feature_set.md).
It replaces the clustering approach
described in [`clustering_method.md`](./clustering_method.md), for the reasons given there — in
short, clustering answers "where is the storm," and this project needs "should the radar fire right
now." This document describes what the track-wise method does, and why it succeeds where clustering
did not.

---

## 1. The idea in one sentence

Instead of detecting storms and then figuring out which ones the satellite observed, this method
starts from the satellite's actual flight path and asks, at every point along it: *what did the
imager see here, and what would the radar have measured here?* Every row is one such moment. No
storm has to be detected for a row to exist.

---

## 2. What it does, step by step

**1. Compute the satellite's real ground track.** Using EarthCARE's actual orbital elements, the
sub-satellite point (the point directly beneath the satellite) is computed once per second across
the full 8-month window. This is the true nadir path — not an approximation.

**2. Fix a decision every 1 minute.** The future RL agent makes a CPR on/off decision every minute
of flight. That is the row rate of the dataset: one row per 1-minute interval across the 8-month
window. The source contains 1,161 missing decision IDs represented by 49 discontinuities; those
gaps are retained and are never bridged.

**3. For each decision, fetch a small patch of weather data.** The G5NR weather simulation updates
every 30 minutes at its finest available resolution (0.0625°, roughly 7 km). For each 1-minute
decision, a small bounding box around the satellite's position at that moment is pulled from the
correct 30-minute weather snapshot — found by simple arithmetic (which 30-minute slot a given
minute falls into), not by matching timestamps, which turned out to be an important distinction
(see §5).

**4. Reduce that patch to a handful of numbers — the diagnostics.** The table contains GEOS-5
Nature Run diagnostics that act as MSI-like and BBR-like proxies, including cloud optical
thickness, cloud fraction, cloud-top temperature, and outgoing longwave radiation. They are not
simulated EarthCARE retrievals: no instrument forward model or retrieval chain has been applied.
The future observation set is tiered; see [`feature_set.md`](./feature_set.md). Nothing about
precipitation enters the always-available context set.

**5. Separately, record what the radar would have measured — the label.** The CPR is a
nadir-pointing instrument with a very narrow footprint, so its label is computed only from the
handful of grid cells directly under the satellite's true sub-satellite track, using the same
per-second track from step 1. This value — precipitation, kept as `prectot_nadir` and
`preccon_nadir` — is stored but never shown to the agent as an input. A second, wider version
(`_swath`) is also stored, representing precipitation anywhere across the imager's full field of
regard, for comparison; the nadir version is the one used to score the agent, because it is the
one the CPR could have actually measured. Using the wider version would overstate what the
instrument sees, the same problem the original clustering approach had at a much larger scale
(§3.3 in the companion document).

**6. Discard the patch and move on.** Nothing is kept in memory beyond the reduced row. This is
what makes the dataset possible at all — see §3.

Repeating this once a minute for 8 months of flight produces one table: 351,639 rows, each one a
complete (diagnostic-proxy context, radar-truth) pair at the intended decision cadence.

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
gap widens as the threshold rises — nadir positives are about 73% of swath positives at
0.1 mm/hr, about half at 0.5–1 mm/hr, and only about a quarter to a third at 5–10 mm/hr.

| threshold | swath positive | nadir positive |
|---|---|---|
| any precipitation | 98.5% | 97.4% |
| ≥ 0.1 mm/hr | 66.3% | 48.6% |
| ≥ 0.5 mm/hr | 40.7% | 21.4% |
| ≥ 1 mm/hr | 28.9% | **13.0%** |
| ≥ 5 mm/hr | 11.1% | 3.5% |
| ≥ 10 mm/hr | 6.9% | 1.8% |

The "any precipitation" row is not usable as a label: the model produces trace drizzle almost
everywhere, so 97.4% of nadir decisions are positive at that threshold.

At the 1 mm/hr operating threshold used for training, **13.0% of decisions are positive** — a
workable minority class for a reward signal, and a realistic one, since it reflects only what the
CPR could physically have observed.

### 4.3 The diagnostic proxies genuinely predict the label

The entire premise of CPR gating only works if cloud properties available to a passive-imager-like
context are informative about precipitation the imager cannot directly measure. This was tested
directly on the finished 8-month dataset, not assumed. Ranking the always-available features by
how well they
separate precipitating from non-precipitating decisions (ROC-AUC, where 0.5 is chance and 1.0 is
perfect separation; "inverted" means lower values indicate precipitation, so the AUC is reported
as 1 − AUC):

| feature | AUC vs. nadir precipitation ≥ 1 mm/hr |
|---|---|
| cloud optical thickness (mean, `tautot_mean`) | **0.815** |
| cloud optical thickness (peak, `tautot_max`) | 0.789 |
| total cloud fraction (`cldtot_mean`) | 0.651 |
| cloud-top temperature (`cldtmp_mean`) | 0.622 (inverted) |
| outgoing longwave radiation (`lwtup_mean`) | 0.598 (inverted) |
| nadir ocean fraction (`ocean_frac_nadir`) | 0.572 |

A logistic regression combining all 12 always-available features (the diagnostic proxies plus
geometry: `solar_hour`, `lat`, `sin_lon`, `cos_lon`, and nadir land/ocean fractions) reaches
**ROC-AUC 0.837** and an average precision of **0.368** against the 13.0% base rate (held-out 30%
split) — a large, genuine lift over chance, using only information the agent is actually allowed
to see. Cloud optical thickness carries the most
signal, which lines up with the physical picture: optically thick cloud is where precipitation
tends to form, so a passive imager measuring cloud thickness is measuring a real precursor of what
the radar would find.

### 4.4 The dataset spans real seasonal variation, not a single repeated snapshot

The window covers June 2005 through January 2006 continuously — 11,731 distinct 30-minute weather
states, not a handful of snapshots replayed many times. The starting month was not arbitrary: all
24 months in the available G5NR record were checked with an area-weighted global average, and
June-July 2005 came out as the wettest consecutive two-month period in the record, so the window
opens on peak convective activity (the Asian monsoon and the Northern Hemisphere ITCZ near their
seasonal peak) and then sweeps forward through a full transition into autumn and winter. That gives
the dataset genuine regime diversity — different seasons, different storm types — rather than one
weather pattern repeated under many different satellite positions.

### 4.5 Data quality was checked and cleaned, not assumed

A small number of readings (710 out of 351,639 rows, 0.20%) were found to contain corrupted values
— a handful of grid cells return an extreme numerical fill value in place of a true reading when
data is missing, which is easy to miss because the value is technically a normal, finite number
rather than an obvious error code. These were identified by checking every feature *and label*
against its physically possible range (for example, cloud fraction can only be between 0 and 1,
and precipitation rate cannot exceed a generous 1 mm/s ceiling) and set to missing rather than
left in the dataset. The extraction code was also corrected so this cannot silently recur on any
future run.

---

## 5. Two things worth knowing if this pipeline is extended

**Match timesteps by position, not by date.** The satellite's ground-track clock and the weather
model's own time axis are on different calendars entirely (the orbital elements are referenced to
a different epoch than the weather archive). Matching them by *finding the nearest matching
timestamp* rather than by *counting how many 30-minute steps have elapsed* is a trap — an earlier
version of this pipeline did exactly that, and it silently mapped every decision in a multi-day run
onto the same single weather snapshot. The fix is to always compute the weather index by direct
arithmetic (decision number → elapsed time → weather-step number), never by nearest-timestamp
lookup across two independently-referenced clocks.

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
| temporal coverage (final run) | a single repeated weather snapshot (original notebook) | 11,731 distinct 30-minute weather states over 8 months |
| memory footprint | full field per snapshot | one small patch per decision, discarded immediately |
| demonstrated learnability | not established on the clustering table | ROC-AUC 0.837, AP 0.368 on always-available features |

---

## 7. Re-extraction (October 2026)

The dataset was extracted a second time. This section explains why, what changed, and what it
affects. Until the downstream steps are rerun on the new extraction, the numbers in §2, §4, and §6
(351,639 rows, 49 discontinuities, ROC-AUC 0.837) describe the old extraction.

### 7.1 How it was previously

Each row is built from two inputs, and each input has its own time.

**Satellite time (TATC + Skyfield).** The orbit is propagated from EarthCARE's TLE, starting at
`startdate = 2025-07-19 15:04 UTC`. Decision *i* is the minute starting at `startdate + i min`.
TATC's `compute_ground_track` produces the MSI footprint for that minute, and Skyfield's
`wgs84.subpoint` produces the per-second nadir points. The row's `time`, `lat`, `lon`, and
`solar_hour` (`UTC hour + lon / 15`) all come from this time. This part was correct: it gives
EarthCARE's real local crossing times of about 14:00 (descending pass) and 02:00 (ascending pass).

**Weather time (G5NR).** The weather snapshot is not chosen by timestamp. It is chosen by index:

```python
g5i = FIELD_START + decision // 30     # one G5NR step (30 min) per 30 decisions
ds[VARS].isel(time=g5i)
```

`FIELD_START` was **773**, which is G5NR's 2005-06-01 **00:00 UTC** snapshot.

The two times advanced at the same rate, but they started at different times of day: the satellite
at 15:04 UTC and the weather at 00:00 UTC. Nothing in the code checked that they matched.

Separately, downloads were handled like this. Each request fetched the weather for 3 decisions and
was retried up to 4 times. If all 4 attempts failed, the request returned an empty list. The block
of 6,000 decisions was then saved as finished anyway. On a restart, saved blocks are skipped, so
the missing decisions were never fetched again.

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

**Missing decisions.** The silently dropped requests left **49 gaps**, totalling **1,161
decisions**. Gap sizes ranged from 3 to 807 decisions. Every gap was a multiple of 3, meaning each
was one or more whole failed requests. Gaps break the 1-minute sequence that RL episodes rely on.

### 7.3 How it was changed

Commits `a826fd8`, `4a06337`, and `b816bc1`, applied to both `extract_trackwise.py` and the
notebook:

1. **`FIELD_START` changed from 773 to 803.**
   ```python
   FIELD_START = 773   # before: 2005-06-01 00:00 UTC
   FIELD_START = 803   # after:  2005-06-01 15:00 UTC
   ```
   803 − 773 = 30 steps = 15 hours. Step 803 was checked against the G5NR server's time axis.
   The satellite side (TLE, `startdate`, TATC, Skyfield) was not changed.
2. **A failed request now returns `None`** instead of an empty list, so a failure can be told apart
   from a request that legitimately had no rows.
3. **Failed requests are retried at block level**, for up to 6 more rounds, waiting 60 s, 2, 4, 8,
   15 and 15 min between rounds (about 45 min total).
4. **An empty response counts as a failure.** During the new run the server once returned zero grid
   cells without an error, for 3 decisions (6492–6494). Fetching the same region again returned
   193 × 94 cells, so it was a server fault. Such responses are now retried.
5. **A block is saved only if all of its requests succeeded.** Otherwise it is logged as
   `INCOMPLETE` and not saved, so rerunning the script fetches it again.
6. **New cache folder** `cache_8mo_jun2005_lst`, so old and new rows cannot mix.

### 7.4 What the solution is

Line up the start of the weather time with the satellite's time of day, and refuse to save
incomplete data.

The selection is still pure index arithmetic, so the §5 rule (never match timestamps across the two
time axes) still holds. Only the starting index moved, so that decision 0 reads the 15:00 UTC
snapshot instead of the 00:00 UTC one. Because both times advance together, they stay aligned for
all 352,800 decisions.

For the gaps, the extractor now separates three cases: a request that succeeded, a request that
failed, and a block that is incomplete. It keeps retrying failures and never saves an incomplete
block.

### 7.5 What it solved

**Time of day.** Each decision now reads the last snapshot taken before the satellite passes over.
The offset is 4–33 minutes, which is the limit set by 30-minute snapshots. Checked on the first
6,000 rows of the new run:

| pass | satellite overhead at | weather before | weather after (median) |
|---|---|---|---|
| descending | ~14:00 | ~22:40 | **13:39** |
| ascending | ~02:00 | ~10:35 | **01:36** |

`solar_hour` now describes both the satellite and the weather in the same row, and daylight-only
features are now in daylight rows.

**Gaps.** Blocks completed so far have 6,000 of 6,000 rows. That includes the block that hit the
empty-response fault, which was refetched in full after the fix. The full run is expected to give
**352,800 rows** (11,760 steps × 30 decisions) with **0 gaps**, instead of 351,639 rows with 49 gaps.

### 7.6 Impact

- **The dataset now represents EarthCARE.** The precipitation the agent is scored on, the class
  balance, and the feature-label relationships are all measured at the mission's real local times.
- **Earlier results no longer apply.** The class-balance table, the feature AUCs, and the ROC-AUC
  0.837 / AP 0.368 in §4.3 were computed on time-shifted weather. They have to be recomputed and
  may change.
- **Everything built from weather values must be regenerated:** the footprint-location repair,
  `solar_hour`, the land/water merge, `lwtup_fore`, the validity flags, the validation reports, and
  the CZML exports. The orbit geometry did not change, so the footprints and the nadir land/water
  fractions themselves come out the same.
- **Row-count checks will fail on purpose.** Several scripts and notebook cell 8 assert 351,639
  rows. They need updating to the new count.
- **RL episodes can be continuous.** With no gaps, episodes no longer have to be split or padded
  around missing minutes. The gap-handling code (for example `lwtup_fore`) still works and will
  just find no gaps.
- **Cost:** one full re-download of about 30 hours. No change to the method, the features, or the
  labels themselves.
