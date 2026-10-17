# 4. Label Geometry: Scoring Against the Radar's Own Footprint

## 4.1 The question

A gating policy must be scored on what the radar could actually measure. The passive sensor almost
always sees a wider area than the radar, so scoring against "precipitation anywhere the passive sensor
sees" rewards the policy for minutes the radar would have missed. How large is that error, and how
does it depend on the radar's geometry?

| mission | radar footprint | passive swath | footprint / swath width |
|---|---|---|---|
| EarthCARE | CPR nadir line (one ~7 km model cell wide) | MSI, 150 km | ~5% |
| TRMM | PR scanned swath, ~245 km | VIRS, ~720 km | ~34% |
| GPM | DPR scanned swath, ~245 km | GMI, ~885 km | ~28% |

The effect should be largest for EarthCARE. For the scanning radars the radar sees a large part of
the passive swath, but a swath-scored policy is still rewarded for precipitation outside it.

## 4.2 Definitions

- **Nadir label:** `prectot_nadir` = maximum precipitation rate over model cells under the 60
  one-second sub-satellite points of the decision.
- **Swath label:** `prectot_swath` = maximum precipitation rate over model cells inside the 150 km
  MSI swath for the same decision.
- **Threshold:** 1 mm/hr = 2.78 × 10⁻⁴ kg m⁻² s⁻¹ (model units are kg m⁻² s⁻¹; × 3600 → mm/hr).
- Because the swath cells are a superset of the nadir cells, swath ≥ nadir always.

## 4.3 EarthCARE results [TO RERUN]

From the **earlier** EarthCARE extraction (finite values only). That extraction paired each orbit
position with weather from the wrong time of day and season and has been deleted; recompute every
number below on the new extraction before use. The size of the effect is expected to hold, since it
comes from geometry, but the values will change:

| target at ≥ 1 mm/hr | finite rows | positive rate |
|---|---|---|
| `prectot_nadir` | 351,615 | **12.97%** |
| `prectot_swath` | 351,613 | **28.93%** |
| `preccon_nadir` (convective) | 351,629 | 1.03% |
| `preccon_swath` (convective) | 351,629 | 4.45% |

- Swath labelling inflates the total-precipitation positive class by **2.2×** and the convective
  positive class by **4.3×**.
- **55.2%** of swath-positive decisions have nadir precipitation below 1 mm/hr: more than half of
  the "positives" a swath-scored policy is rewarded for are minutes the radar could not have
  measured.
- No decision is nadir-positive but swath-negative (as expected from the superset geometry).
- Mean rate: nadir 2.8 × 10⁻⁴, swath 1.03 × 10⁻³ kg m⁻² s⁻¹ (3.7× higher).

## 4.4 TRMM and GPM [TO RUN]

The same comparison for the scanning radars: label = maximum over cells in the PR or DPR swath vs.
maximum over cells in the VIRS or GMI swath. For GPM, the comparison applies to the precipitation
label used in the secondary analysis; the primary GPM label measures the radar's added value
(`05_baselines.md` §5.8).

## 4.5 Experiments still needed [TO RUN]

The positive-rate gap is established. Its effect on policy evaluation is not. Run the 2 × 2 grid in
`07_experiments.md` (E4):

| | scored on nadir | scored on swath |
|---|---|---|
| **trained on nadir** | honest result | — |
| **trained on swath** | what a swath-trained policy really delivers | what a swath study would report |

The headline number is the gap between the bottom-right cell (what the study would claim) and the
bottom-left cell (what the radar would actually get). Report it as capture@d and as AUC.

Also report the same comparison for `preccon` (convective), where the inflation is larger.

## 4.6 Connection to the earlier clustering design

The same error appeared at a larger scale in the storm-object design: a pass grazing one corner of a
merged object was credited with up to 39× more area than it could image
([`../clustering_method.md`](../clustering_method.md) §3.3). The track-wise nadir label removes both
problems by scoring only the cells the radar could have sampled.

## 4.7 Why this matters beyond this project (discussion paragraph)

Any targeting study for a nadir active sensor (radar or lidar) that uses imager-scale or gridded
truth will overstate how often there is something to measure, and therefore overstate both the
achievable capture and the advantage of the learned policy over simple baselines. The fix is cheap —
compute the label from the instrument footprint — and the dataset ships both labels so the effect can
be reproduced.

---

### Reproduction snippet

```python
import pandas as pd
df = pd.read_parquet("trackwise_decisions.parquet")
thr = 1 / 3600  # 1 mm/hr in kg m-2 s-1
for c in ["prectot_nadir", "prectot_swath", "preccon_nadir", "preccon_swath"]:
    v = df[c].dropna()
    print(c, len(v), round((v >= thr).mean(), 4))
m = df.prectot_nadir.notna() & df.prectot_swath.notna()
n = (df.prectot_nadir >= thr)[m]
s = (df.prectot_swath >= thr)[m]
print("swath-pos & nadir-neg / swath-pos:", round((s & ~n).sum() / s.sum(), 4))
```
