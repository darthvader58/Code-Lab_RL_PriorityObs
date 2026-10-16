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

**Satellite time (TATC + Skyfield).** The orbit was propagated from an EarthCARE TLE with epoch
2025-07-19 08:11 UTC, starting at `startdate = 2025-07-19 15:04 UTC`. Decision *i* is the minute
starting at `startdate + i min`. TATC's `compute_ground_track` produces the MSI footprint for that
minute, and Skyfield's `wgs84.subpoint` produces the per-second nadir points. The row's `time`,
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

**Missing decisions.** The silently dropped requests left **49 gaps**, totalling **1,161
decisions**. Gap sizes ranged from 3 to 807 decisions. Every gap was a multiple of 3, meaning each
was one or more whole failed requests. Gaps break the 1-minute sequence that RL episodes rely on.

**Problems found while re-extracting.** The first fix (moving `FIELD_START` to 803, §7.3) was
running when a review and the run itself turned up four more issues:

- **4-minute boundary error.** Step 803 is 15:00 UTC, but the orbit started at 15:04. So the
  satellite at 15:30–15:33 still read the 15:00 snapshot although 15:30 was available. 4 of every
  30 decisions (13%) used a snapshot 30 minutes older than necessary, and the offset was 4–33 min
  instead of 0–29.
- **Repeat-cycle risk.** Checked directly: for the old TLE, TATC found no repeat cycle, and TATC and
  Skyfield positions agreed to within 0.1 m at 8 dates across the window. So no data was affected,
  but a different TLE could have split the two trajectories without any error.
- **Empty responses.** The server once returned zero grid cells without an error, for 3 decisions
  (6492–6494). Fetching the same region again returned 193 × 94 cells, so it was a server fault,
  but the code treated it as a success and saved a 3-decision gap.
- **Hung downloads.** When the VPN disconnected mid-run, 12 of 18 workers were left on connections
  through an address that no longer existed. libcurl waited on them indefinitely and ignored
  `HTTP.TIMEOUT`, so a block stalled for over 90 minutes with no error and no retry.

### 7.3 How it was changed

Commits `a826fd8`, `4a06337`, `b816bc1`, `c39fc32`, and `9dc8377`, applied to
`extract_trackwise.py` and the notebook (the TLE and `startdate` also in the `task_outputs` scripts
that re-propagate the orbit):

1. **`FIELD_START` changed from 773 to 803.**
   ```python
   FIELD_START = 773   # before: 2005-06-01 00:00 UTC
   FIELD_START = 803   # after:  2005-06-01 15:00 UTC
   ```
   803 − 773 = 30 steps = 15 hours. Step 803 was checked against the G5NR server's time axis.
2. **`startdate` changed to a 15:00 UTC start, with a current TLE.**
   ```python
   startdate = datetime(2025, 7, 19, 15, 4, tzinfo=timezone.utc)   # before
   startdate = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)  # after
   ```
   The TLE was replaced with the latest CelesTrak element set for EarthCARE (NORAD 59908), epoch
   2026-10-14 21:54 UTC. `startdate` is the first 15:00 UTC after that epoch, so the orbit is
   propagated forward from fresh elements and starts at the same time of day as step 803. The
   orbit's calendar date does not need to match the weather's; only the time of day does.
3. **TATC's repeat cycle is switched off:** `config.rc.repeat_cycle_for_orbit_track = False`. With
   it off, TATC propagates the TLE the same way Skyfield does.
4. **A failed request now returns `None`** instead of an empty list, so a failure can be told apart
   from a request that legitimately had no rows.
5. **Failed requests are retried at block level**, for up to 6 more rounds, waiting 60 s, 2, 4, 8,
   15 and 15 min between rounds (about 45 min total).
6. **An empty response counts as a failure** and is retried.
7. **Each task has a 15-minute time limit** (`TASK_TIMEOUT = 900`). A normal task takes about 15 s.
   If any task exceeds the limit, the worker pool is killed and all of that call's tasks count as
   failed and go through the retry rounds.
8. **A block is saved only if all of its requests succeeded.** Otherwise it is logged as
   `INCOMPLETE` and not saved, so rerunning the script fetches it again.
9. **New cache folder** `cache_8mo_jun2005_aligned`. All data from earlier attempts was deleted, and
   the extraction was restarted from the first block.

### 7.4 What the solution is

Make both times start at the same time of day, 15:00 UTC, and refuse to save incomplete data.

The weather selection is still pure index arithmetic, so the §5 rule (never match timestamps across
the two time axes) still holds. With the orbit starting at 15:00 UTC and decision 0 reading the
15:00 UTC snapshot, the 30 decisions in each 30-minute block all read the snapshot from the start
of their own half hour. Both times advance together, so they stay aligned for all 352,800
decisions. Fixing the alignment through `startdate` rather than by adding an offset inside the index
formula keeps the rule "decision *i* belongs to step *i* // 30", which the block, field, and
request grouping all depend on.

Footprints and nadir points come from one propagation of one TLE, because the TATC repeat cycle is
off.

For downloads, the extractor separates three cases: a request that succeeded, a request that failed
(including empty responses and tasks over the time limit), and a block that is incomplete. It keeps
retrying failures and never saves an incomplete block.

### 7.5 What it solved

**Time of day.** Every decision now reads the latest snapshot at or before the satellite's time.
The offset is 0–29 minutes for all 352,800 decisions, the minimum possible with 30-minute snapshots.
Computed over the full window from the new orbit and the index mapping (latitudes within ±60°):

| pass | satellite overhead at (median) | weather before | weather after (median) |
|---|---|---|---|
| descending | 13:56 | ~22:40 | **13:43** |
| ascending | 01:56 | ~10:35 | **01:43** |

With the new TLE, the descending node stays at 13:56–13:58 and the ascending node at 01:56–01:58
local time across the whole 8-month window.

`solar_hour` now describes both the satellite and the weather in the same row, and daylight-only
features are now in daylight rows.

**Trajectory.** With the repeat cycle off, TATC and Skyfield nadir positions agree to floating-point
precision at every date checked, and footprint centroids stay 40–42 km from nadir (the 5.76° MSI
roll) at the start, middle, and end of the window.

**Gaps.** Failed, empty, and hung downloads are all retried, and no incomplete block is saved. The
full run is expected to give **352,800 rows** (11,760 steps × 30 decisions) with **0 gaps**,
instead of 351,639 rows with 49 gaps.

### 7.6 Impact

- **The dataset now represents EarthCARE.** The precipitation the agent is scored on, the class
  balance, and the feature-label relationships are all measured at the mission's real local times,
  from a current orbit.
- **Earlier results no longer apply.** The class-balance table, the feature AUCs, and the ROC-AUC
  0.837 / AP 0.368 in §4.3 were computed on time-shifted weather. They have to be recomputed and
  may change.
- **Everything downstream must be regenerated, including geometry.** The new TLE and `startdate`
  change where the satellite is at each decision. So the footprints, the nadir land/water fractions,
  the footprint-location repair, `solar_hour`, the land/water merge, `lwtup_fore`, the validity
  flags, the validation reports, and the CZML exports all have to be rebuilt from the new rows. The
  orbit dates in the `time` column now run from 2026-10-15 to 2027-06-17.
- **Row-count checks will fail on purpose.** Several scripts and notebook cell 8 assert 351,639
  rows. They need updating to the new count.
- **RL episodes can be continuous.** With no gaps, episodes no longer have to be split or padded
  around missing minutes. The gap-handling code (for example `lwtup_fore`) still works and will
  just find no gaps.
- **Cost:** one full re-download of about 30 hours, plus the partial runs discarded along the way.
  No change to the method, the features, or the labels themselves.
