# 6. Onboard Feasibility

## 6.1 The objection

"Current satellites do not have enough compute to take one sensor's data and run an RL agent to
decide when to fire another." This section separates that into two costs and measures each.

1. **Policy cost:** evaluating the trained policy once per decision.
2. **Feature cost:** turning raw imager pixels into the policy's input features.

Training happens on the ground. Only the frozen policy runs onboard.

## 6.2 Policy cost

One decision per minute, 12 inputs. Fill in the measured values; the estimates below are
arithmetic, not benchmarks.

| policy | operations per decision (estimate) | parameters | memory (float32) |
|---|---|---|---|
| logistic regression | 12 multiply-adds + 1 sigmoid | 13 | ~52 B |
| gradient-boosted trees (200 trees, depth 6) | ~1,200 comparisons | ~25k node values | ~100–200 kB |
| MLP 12-64-64-1 | ~5,000 multiply-adds | ~5,000 | ~20 kB |

A radiation-hardened LEON-class flight processor executes on the order of tens to hundreds of
millions of instructions per second. Even the MLP is well under a millisecond per decision, against a
60-second decision interval. **[TO RUN]:** time each policy on a constrained CPU (e.g., a single core
with frequency capped, or a Raspberry Pi–class board) and report wall time per decision. State
clearly that this is a proxy, not flight hardware.

**Claim for the paper:** running the policy is not the bottleneck. The bottleneck is the inputs.

## 6.3 Feature cost: three tiers (EarthCARE; TRMM and GPM in §6.7)

| tier | features | needs onboard retrieval? | works at night? |
|---|---|---|---|
| **A — full** | all 12 always-available: `cldtot_mean`, `cldtmp_mean`, `tautot_mean`, `tautot_max`, `lwtup_mean`, `lwtup_fore`, `solar_hour`, `lat`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` | yes (cloud fraction, optical depth, cloud-top temperature) | no (`tautot` needs daylight) |
| **B — cheap** | `cldtmp_mean`, `lwtup_mean`, `lwtup_fore` + Tier C | approximately no: thermal-infrared quantities close to brightness temperatures | yes |
| **C — geometry** | `lat`, `solar_hour`, `sin_lon`, `cos_lon`, `land_frac_nadir`, `ocean_frac_nadir` | none (orbit propagation + a static land mask) | yes |

Honesty notes for the paper:
- `cldtmp_mean` is itself a model cloud-top temperature, not a raw radiance. It is the closest
  available stand-in for an MSI thermal-infrared brightness temperature, which is cheap to compute
  from calibrated counts. Say this explicitly; do not call Tier B "raw radiances."
- `lwtup_*` is broadband outgoing longwave from the model, a stand-in for a BBR-like or TIR-window
  signal.
- Tier C needs no imager and is the floor: anything the imager adds must beat it.

## 6.4 Day vs. night

`tautot_mean` and `tautot_max` are daylight-only for a real imager. Define day by solar zenith angle
(or, as a first pass, `solar_hour` between 06 and 18 local) and report every tier separately for:

- day decisions (Tier A valid),
- night decisions (Tier A must drop `tautot*`; Tier B and C unchanged),
- all decisions with a day/night switch (Tier A by day, Tier B by night).

The switched policy is the realistic onboard configuration.

## 6.5 The result to report [TO RUN]

**Skill retention** = capture@d (Tier B) ÷ capture@d (Tier A), at each budget *d*, overall and
at night.

If Tier B retains most of Tier A's capture, the paper can state that useful radar gating does not
require onboard cloud retrievals, only thermal-infrared quantities and a microcontroller-sized model.
If it does not, the paper states the opposite: the gating value depends on onboard retrievals, which
sets a hardware requirement for future missions. Either result is publishable.

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
