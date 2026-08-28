# The Track-Wise Method: What It Does and Why It Works

**Applies to:** `geos5data_0.0625deg_30mn_trackwise.ipynb` and its extraction script
`extract_trackwise.py`. This is the dataset used for training. It replaces the clustering approach
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

**2. Fix a decision every 1 minute.** The RL agent makes a CPR on/off decision every minute of
flight. That is the row rate of the dataset: one row per 1-minute interval, for the entire 8
months, with no gaps and no sampling — every single decision the agent could ever face is
represented.

**3. For each decision, fetch a small patch of weather data.** The G5NR weather simulation updates
every 30 minutes at its finest available resolution (0.0625°, roughly 7 km). For each 1-minute
decision, a small bounding box around the satellite's position at that moment is pulled from the
correct 30-minute weather snapshot — found by simple arithmetic (which 30-minute slot a given
minute falls into), not by matching timestamps, which turned out to be an important distinction
(see §5).

**4. Reduce that patch to a handful of numbers — the features.** Everything MSI could plausibly
measure — cloud optical thickness, cloud fraction (total, high, mid, low), cloud-top temperature,
outgoing longwave radiation — is averaged over the imager's field of view at that moment. These
averages become the agent's observation. Nothing about precipitation enters this set.

**5. Separately, record what the radar would have measured — the label.** The CPR is a
nadir-pointing instrument with a very narrow footprint, so its label is computed only from the
handful of grid cells directly under the satellite's true sub-satellite track, using the same
per-second track from step 1. This value — precipitation, kept as `prectot_nadir` and
`preccon_nadir` — is stored but never shown to the agent as an input. A second, wider version
(`_swath`) is also stored, representing precipitation anywhere across the imager's full field of
regard, for comparison; the nadir version is the one used to score the agent, because it is the
one the CPR could have actually measured. Using the wider version would overstate what the
instrument sees, the same problem the original clustering approach had at a much larger scale
(§4 in the companion document).

**6. Discard the patch and move on.** Nothing is kept in memory beyond the reduced row. This is
what makes the dataset possible at all — see §3.

Repeating this once a minute for 8 months of flight produces one table: 351,639 rows, each one a
complete (MSI-visible conditions, radar truth) pair, at exactly the cadence the agent will actually
operate at.

---

## 3. Why the data has to be fetched this way

A tempting alternative is to just download the whole weather grid for the whole period and query
it locally. At the resolution and cadence this project needs, that is not possible:

$$\underbrace{2{,}881 \times 5{,}760}_{\text{cells at 0.0625°}} \times \underbrace{13}_{\text{variables}} \times \underbrace{11{,}760}_{\text{30-min steps, 8 months}} \approx 11.7\ \text{TB}.$$

The satellite only ever flies over a thin ribbon of the planet at any moment, so almost all of that
11.7 TB would be weather data the agent never needs, over parts of the globe it isn't looking at.
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
to the true nadir track corrects this directly, and the effect is large: at every precipitation
threshold tested, the nadir-based positive rate is roughly half the swath-based one.

| threshold | swath positive | nadir positive |
|---|---|---|
| any precipitation | 98.5% | 97.4% |
| ≥ 0.1 mm/hr | 66.3% | 48.6% |
| ≥ 0.5 mm/hr | 40.7% | 21.5% |
| ≥ 1 mm/hr | 28.9% | **13.0%** |
| ≥ 5 mm/hr | 11.1% | 3.5% |
| ≥ 10 mm/hr | 6.9% | 1.9% |

At the 1 mm/hr operating threshold used for training, **13.0% of decisions are positive** — a
workable minority class for a reward signal, and a realistic one, since it reflects only what the
CPR could physically have observed.

### 4.3 The MSI-visible features genuinely predict the label

The entire premise of CPR gating only works if cloud properties visible to a passive imager are
actually informative about precipitation the imager cannot directly measure. This was tested
directly on the finished 8-month dataset, not assumed. Ranking individual features by how well they
separate precipitating from non-precipitating decisions (ROC-AUC, where 0.5 is chance and 1.0 is
perfect separation):

| feature | AUC vs. nadir precipitation ≥ 1 mm/hr |
|---|---|
| cloud optical thickness (mean) | **0.815** |
| cloud optical thickness (peak) | 0.789 |
| low-cloud optical thickness | 0.734 |
| high-cloud optical thickness | 0.711 |
| mid-level cloud fraction | 0.694 |
| total cloud fraction | 0.651 |

A logistic regression combining all eleven MSI-visible features reaches **ROC-AUC 0.838** and an
average precision of 0.378 against the 13.0% base rate — a large, genuine lift over chance, using
only information the agent is actually allowed to see. Cloud optical thickness carries the most
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

A small number of readings (640 out of 351,639 rows, 0.18%) were found to contain corrupted values
— a handful of grid cells return an extreme numerical fill value in place of a true reading when
data is missing, which is easy to miss because the value is technically a normal, finite number
rather than an obvious error code. These were identified by checking every feature against its
physically possible range (for example, cloud fraction can only be between 0 and 1) and set to
missing rather than left in the dataset. The extraction code was also corrected so this cannot
silently recur on any future run.

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
| demonstrated learnability | not established on the clustering table | ROC-AUC 0.838, AP 0.378 on real features |
