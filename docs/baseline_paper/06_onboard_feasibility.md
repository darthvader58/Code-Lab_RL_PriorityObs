# 6. Onboard Feasibility

## 6.1 The objection

"Current satellites do not have enough compute to take one sensor's data and run an RL agent to
decide when to fire another." This section separates that into two costs and measures each.

1. **Policy cost:** evaluating the trained policy once per decision.
2. **Feature cost:** turning raw imager pixels into the policy's input features.

Training happens on the ground. Only the frozen policy runs onboard.

## 6.2 Policy cost

One decision per minute, 13 inputs (Tier A). Measured on EarthCARE (E9):

| policy | operations per decision | parameters | serialized size | wall time per decision (Python, laptop CPU) |
|---|---|---|---|---|
| P2 geometry LR | ~7 multiply-adds + 1 sigmoid | 8 | 2.8 kB | 0.85 ms |
| P5 logistic regression | ~15 multiply-adds + 1 sigmoid | 16 | 3.3 kB | 0.83 ms |
| P7 MLP 15-64-64-1 | ~5,120 multiply-adds | 5,249 | 134 kB | 1.30 ms |
| P6 LightGBM, 300 trees (mean depth 11.2) | ~3,350 comparisons | 18,300 | 1.05 MB | 1.06 ms |

The serialized sizes include the scikit-learn/LightGBM object overhead; the raw parameters are
smaller (16 floats for logistic regression, ~21 kB for the MLP in float32). The wall times are
dominated by Python call overhead, not arithmetic, and vary by about ±0.3 ms between runs on the same
laptop; the four policies are all about 1 ms per decision against a 60-second decision interval. A
radiation-hardened LEON-class flight processor executes on the order of tens to hundreds of millions
of instructions per second, so even the largest policy (~5,000 multiply-adds) needs well under a
millisecond of arithmetic per decision. **Not measured here (Limitation 11):** timing on
constrained or flight-representative hardware, such as a Raspberry Pi–class board or a LEON-class
processor. The onboard claim above is an estimate from operation counts; a real-life use case needs
that measurement (future work 7).

LightGBM's extra 6.1 capture points over logistic regression cost 320× the model size; the MLP gets
within 1.4 points of LightGBM at 1/8 of its size.

**Claim for the paper:** running the policy is not the bottleneck. The bottleneck is the inputs.

## 6.3 Feature cost: three tiers (EarthCARE; TRMM and GPM in §6.7)

| tier | features | needs onboard retrieval? | works at night? |
|---|---|---|---|
| **A — full** | 13: `cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`, `lwtup_fore` + Tier C | yes (cloud fraction, optical depth, cloud-top temperature) | no (`tautot` needs daylight) |
| **A − τ** | Tier A without `tautot_mean`, `tautot_max` (11) | yes (cloud fraction, cloud-top temperature) | yes |
| **B — cheap** | `cldtmp_mean`, `lwtup_mean`, `lwtup_fore` + Tier C (10) | approximately no: thermal-infrared quantities close to brightness temperatures | yes |
| **C — geometry** | `lat`, `sin_sh`, `cos_sh` (sine/cosine of `solar_hour`), `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` (7) | none (orbit propagation + a static land mask) | yes |

Honesty notes for the paper:
- `cldtmp_mean` is itself a model cloud-top temperature, not a raw radiance. It is the closest
  available stand-in for an MSI thermal-infrared brightness temperature, which is cheap to compute
  from calibrated counts. Say this explicitly; do not call Tier B "raw radiances."
- `lwtup_*` is broadband outgoing longwave from the model, a stand-in for a BBR-like or TIR-window
  signal.
- Tier C needs no imager and is the floor: anything the imager adds must beat it.

## 6.4 Day vs. night

`tautot_mean` and `tautot_max` are daylight-only for a real imager. Day is defined as `solar_hour`
between 06 and 18 local (a first pass; solar zenith angle would be more exact), and every tier is
reported separately for:

- day decisions (Tier A valid),
- night decisions (Tier A must drop `tautot*`; Tier B and C unchanged),
- all decisions with a day/night switch (each tier by day, the same tier without `tautot*` by night).

The switched policy is the realistic onboard configuration.

## 6.5 Result (EarthCARE, E5)

**Skill retention** = capture@d (tier, switched day/night) ÷ capture@d (Tier A, switched), at 20%
duty with LightGBM:

| tier | day capture@20% | night capture@20% | switched capture@20% | retention vs. A |
|---|---|---|---|---|
| A (full) | 67.6% | n/a (`tautot` invalid) | 58.5% | 100% |
| A − τ | 55.1% | 49.8% | 52.1% | 89.2% |
| B (cheap) | 55.9% | 49.8% | 52.7% | 90.2% |
| C (geometry) | 33.2% | 33.1% | 33.2% | 56.7% |

(The switched values are lower than the day-only ones because night minutes have no optical depth.)

**Finding:** Tier B retains 90% of Tier A's switched capture, and at night it matches the richer
A − τ tier exactly (49.8%). Useful radar gating on EarthCARE therefore does **not** require onboard
cloud retrievals: thermal-infrared-like quantities plus orbit geometry and a model of 3 kB – 1 MB are
enough. Optical depth adds ~12 points by day (67.6% vs. 55.9%), so a daytime optical-depth retrieval
is worth having but not required. Geometry alone (Tier C) keeps only 57%, so the passive sensor
contributes most of the skill.

## 6.6 Latency budget (one paragraph in the paper)

At ~7 km/s ground speed, MSI and VIRS see the scene essentially when the radar does. GMI's footprint
lies ahead of the spacecraft, so on GPM the passive view of a scene can precede the radar's. For the
gate to act on the current minute, features + policy must complete within seconds. Policy time is
negligible (6.2); feature time depends on the tier (6.3). `lwtup_fore` is the only feature that
looks ahead along track, and it is a proxy (see `03_dataset.md` §3.5).

## 6.7 Feature tiers for TRMM and GPM

The policy cost (§6.2) is the same for every mission. The feature cost is not.

| mission | Tier A (full) | Tier B (cheap) | Tier C (geometry) | works at night? |
|---|---|---|---|---|
| EarthCARE | MSI-style cloud proxies + geometry (§6.3) | thermal-infrared-like proxies + geometry | geometry | Tier A no, B and C yes |
| TRMM | VIRS-style cloud proxies + geometry; optionally + TMI | thermal-infrared-like proxies + geometry | geometry | Tier A no, B and C yes |
| GPM | all 13 GMI channels + geometry | a few GMI channels (e.g., 37 and 89 GHz) + geometry | geometry | yes, all tiers |

- **TRMM** follows EarthCARE closely. Because TRMM samples all local times, its day/night split has
  data at every hour and latitude, which EarthCARE's two fixed local times cannot provide.
- **GPM** needs no onboard retrieval in any tier: brightness temperatures come straight from
  calibrated counts, and microwave works day and night through cloud. The cost question for GPM is
  therefore not "can the features be computed onboard?" but "how many channels does the gate need?"
- **Tier C** (orbit position, surface type, local time) is the floor in every mission; anything the
  passive sensor adds must beat it.
