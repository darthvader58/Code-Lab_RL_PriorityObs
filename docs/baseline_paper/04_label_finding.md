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

## 4.3 EarthCARE results

From the EarthCARE dataset (352,800 decisions, all finite):

| target at ≥ 1 mm/hr | positive rate |
|---|---|
| `prectot_nadir` | **13.65%** |
| `prectot_swath` | **29.99%** |
| `preccon_nadir` (convective) | 1.16% |
| `preccon_swath` (convective) | 5.04% |

- Swath labelling inflates the total-precipitation positive class by **2.2×** and the convective
  positive class by **4.3×**.
- **54.5%** of swath-positive decisions have nadir precipitation below 1 mm/hr: more than half of
  the "positives" a swath-scored policy is rewarded for are minutes the radar could not have
  measured.
- No decision is nadir-positive but swath-negative (as expected from the superset geometry).
- Mean rate: nadir 3.0 × 10⁻⁴, swath 1.15 × 10⁻³ kg m⁻² s⁻¹ (3.8× higher).
- The gap widens with the threshold: nadir positives are 74% of swath positives at 0.1 mm/hr, 46%
  at 1 mm/hr, and 26% at 10 mm/hr (`../EarthCare_trackwise_method.md` §4.2).

## 4.4 TRMM and GPM [TO RUN]

The same comparison for the scanning radars: label = maximum over cells in the PR or DPR swath vs.
maximum over cells in the VIRS or GMI swath. For GPM, the comparison applies to the precipitation
label used in the secondary analysis; the primary GPM label measures the radar's added value
(`05_baselines.md` §5.8).

## 4.5 Effect on policy evaluation (E4, EarthCARE)

LightGBM on Tier A, time-blocked test (Nov 2005 – Jan 2006), capture at 20% duty and AUC:

| trained on ↓ / scored on → | nadir (what the radar measures) | swath |
|---|---|---|
| **nadir** | **66.4%** (AUC 0.893) | 51.1% (AUC 0.897) |
| **swath** | 64.7% (AUC 0.890) | 50.7% (AUC 0.899) |

Two results:

- **Ranking barely depends on the label.** AUC is 0.89–0.90 in all four cells, and a policy trained
  on swath rain loses only 1.7 points of nadir capture against one trained on nadir rain. The
  passive features point to the same scenes either way.
- **The reported capture depends a lot on the label.** For the same 20% budget, scoring on the swath
  label gives about 51% instead of 66%, because swath rain is 2.2× more common, so a 20% budget can
  cover a smaller share of it. A swath-scored study would report a different, and not comparable,
  number from the one the radar would deliver. The headline numbers must use the radar's own
  footprint.
- **What the swath label does inflate is precision.** With the radar on 20% of the time, about 71%
  of radar-on minutes contain swath rain but only about 42% contain rain the radar can measure
  (capture × base rate ÷ duty: 0.507 × 0.279 / 0.2 and 0.664 × 0.128 / 0.2). A swath-scored study
  would claim the radar finds something worth measuring on most of its on-time; in fact it does so
  on fewer than half.

The convective (`preccon`) version of the grid is not run yet; E7 reports the convective nadir label
alone (LightGBM AUC 0.957, capture 30.4% at d = its 1.2% base rate).

## 4.6 Connection to the earlier clustering design

The same error appeared at a larger scale in the storm-object design: a pass grazing one corner of a
merged object was credited with up to 39× more area than it could image
([`../clustering_method.md`](../clustering_method.md) §3.3). The track-wise nadir label removes both
problems by scoring only the cells the radar could have sampled.

## 4.7 Why this matters beyond this project (discussion paragraph)

Any targeting study for a nadir active sensor (radar or lidar) that uses imager-scale or gridded
truth will overstate how often there is something to measure: on EarthCARE, 2.2× more rainy
minutes, and a precision of ~71% instead of ~42% at a 20% budget. Capture under a fixed budget moves
the other way (51% vs. 66%), so swath-scored and footprint-scored numbers are not comparable in
either direction. The fix is cheap — compute the label from the instrument footprint — and the
dataset ships both labels so the effect can be reproduced.

---

### Reproduction snippet

```python
import pandas as pd
df = pd.read_parquet("missions/earthcare/data/EarthCare_trackwise_decisions.parquet")
thr = 1 / 3600  # 1 mm/hr in kg m-2 s-1
for c in ["prectot_nadir", "prectot_swath", "preccon_nadir", "preccon_swath"]:
    v = df[c].dropna()
    print(c, len(v), round((v >= thr).mean(), 4))
m = df.prectot_nadir.notna() & df.prectot_swath.notna()
n = (df.prectot_nadir >= thr)[m]
s = (df.prectot_swath >= thr)[m]
print("swath-pos & nadir-neg / swath-pos:", round((s & ~n).sum() / s.sum(), 4))
```
