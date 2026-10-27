#!/usr/bin/env python3
"""Build a mission's analysis notebook (cells only; execute it with the trackwise-py kernel).

    python3 task_outputs/build_notebook.py --mission earthcare
    jupyter nbconvert --to notebook --execute --inplace \\
        --ExecutePreprocessor.kernel_name=trackwise-py missions/earthcare/EarthCare_trackwise.ipynb

Writes missions/<mission>/<Prefix>_trackwise.ipynb. Every number in the notebook's text is computed
from the mission's dataset and results when it runs; only the mission description below is fixed.
"""
import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trackwise_missions  # noqa: E402

# Mission description: the parts of the text that are facts about the mission, not results.
TEXT = {
    "earthcare": {
        "question": "should the Cloud Profiling Radar (CPR) be switched on this minute, decided only "
                    "from what the passive Multi-Spectral Imager (MSI) and the orbit geometry can see?",
        "orbit": "EarthCARE's position every second, from a set of real TLEs (NORAD 59908) covering "
                 "2025-06-01 → 2026-02-01. The TLE closest in time is used for each moment, so orbit "
                 "manoeuvres are followed.",
        "alignment": "The orbit runs in 2025, the weather is from 2005. Each orbit minute is paired with "
                     "the G5NR snapshot at the **same day of year and the same UTC time** (orbit "
                     "2025-06-01 15:00 ↔ G5NR 2005-06-01 15:00, step 803), so the season and the local "
                     "solar time are both correct.",
        "features": "MSI-like cloud diagnostics averaged over the 150 km MSI swath under the satellite "
                    "during that minute (cloud fraction, cloud-top temperature, optical depth, outgoing "
                    "longwave).",
        "targets": "Precipitation, ice and liquid water: the maximum along the **nadir** track (what the "
                   "CPR actually measures) and over the whole **swath**.",
        "enrichment": "Land/water fraction under the track, cyclic longitude, solar hour, next-minute "
                      "longwave (`lwtup_fore`), validity flags.",
        "note": "One block of the extraction (block 13) received a corrupted server response that put "
                "precipitation of up to 60,030 kg m⁻² s⁻¹ in 3 rows over Antarctica. The block was "
                "re-downloaded, and the re-run matched the original in every value except those 3 rows. "
                "The range check in §2 verifies that no physically impossible values remain.",
        "label": "**\"precipitation ≥ 1 mm/hr at nadir\"**, i.e. rain the CPR would actually see. The "
                 "swath column counts rain anywhere in the 150 km MSI swath, much of which the "
                 "nadir-pointing CPR never flies over.",
        "footprint": "at nadir",
        "e4": "The CPR only measures the nadir line, so a policy credited with swath rain is credited "
              "with rain the radar never sees.",
        "tiers": "| A | everything (cloud fraction, cloud-top temperature, optical depth, longwave and its "
                 "next-minute value, geometry) |\n"
                 "| A_notau | A without optical depth (what is available at night) |\n"
                 "| B | thermal only: cloud-top temperature and longwave, plus geometry |\n"
                 "| C | geometry only |",
        "night": "Optical depth (`tautot`) needs reflected sunlight, so at night the MSI has only thermal "
                 "channels.",
    },
    "trmm": {
        "question": "should the Precipitation Radar (PR) be switched on this minute, decided only from "
                    "what the passive Visible and Infrared Scanner (VIRS) and the orbit geometry can see?",
        "orbit": "TRMM's position every second, from a set of real TLEs (NORAD 25063) covering "
                 "2005-06-01 → 2006-02-01: a 35° precessing orbit at ~400 km that samples every local "
                 "time. The TLE closest in time is used for each moment.",
        "alignment": "TRMM flew during G5NR's own period, so the orbit and the weather share the "
                     "**calendar date and UTC time** (orbit 2005-06-01 15:00 = G5NR 2005-06-01 15:00, "
                     "step 803).",
        "features": "VIRS-like cloud diagnostics averaged over the 833 km VIRS swath under the satellite "
                    "during that minute (cloud fraction, cloud-top temperature, optical depth, and "
                    "outgoing longwave as a stand-in for the VIRS thermal-infrared window; TRMM had no "
                    "working broadband radiometer in 2005).",
        "targets": "Total and convective precipitation: the maximum over the 247 km **PR swath** (what the "
                   "PR measures, the label), its mean over the PR swath (the weight for mass-capture), "
                   "and for comparison the maximum over the **VIRS swath** and along the **nadir** line.",
        "enrichment": "Land/water fraction under the track, cyclic longitude, solar hour, validity flags. "
                      "No forward-view feature: VIRS is a cross-track scanner.",
        "note": "The extraction refuses to save a block unless every request succeeded, and any footprint "
                "without a grid cell inside it stops the run. The range check in §2 verifies that no "
                "physically impossible values reached the dataset.",
        "label": "**\"precipitation ≥ 1 mm/hr anywhere in the PR swath\"**, i.e. rain the scanning PR "
                 "would measure. The swath column counts rain anywhere in the 833 km VIRS swath, most of "
                 "which lies outside the PR's 247 km; the nadir column shows what a nadir-only radar "
                 "(like EarthCARE's CPR) would see on the same track.",
        "footprint": "in the PR swath",
        "e4": "The PR measures a 247 km swath inside the 833 km VIRS swath, so a policy credited with "
              "VIRS-swath rain is credited with rain the radar never scans. The nadir line shows how a "
              "nadir-only radar on the same orbit would score.",
        "tiers": "| A | everything (cloud fraction, cloud-top temperature, optical depth, longwave, geometry) |\n"
                 "| A_notau | A without optical depth (what is available at night) |\n"
                 "| B | thermal only: cloud-top temperature and longwave, plus geometry |\n"
                 "| C | geometry only |",
        "night": "Optical depth (`tautot`) needs reflected sunlight, so at night VIRS has only its "
                 "thermal channels.",
    },
    "fy3g": {
        "question": "should the Precipitation Measurement Radar (PMR) be switched on this minute, decided "
                    "only from what the passive Medium Resolution Spectral Imager (MERSI-RM) and the "
                    "orbit geometry can see?",
        "orbit": "FY-3G's position every second, from a set of real TLEs (NORAD 56232) covering "
                 "2025-06-01 → 2026-02-01: a 50° precessing orbit at ~407 km that samples every local "
                 "time. The TLE closest in time is used for each moment, so the two orbit raises in the "
                 "window (August and September 2025) are followed.",
        "alignment": "The orbit runs in 2025, the weather is from 2005. Each orbit minute is paired with "
                     "the G5NR snapshot at the **same day of year and the same UTC time** (orbit "
                     "2025-06-01 15:00 ↔ G5NR 2005-06-01 15:00, step 803), as for EarthCARE, so the "
                     "season and the local solar time are both correct.",
        "features": "MERSI-RM-like cloud diagnostics averaged over the 1000 km MERSI-RM swath under the "
                    "satellite during that minute (cloud fraction, cloud-top temperature, optical depth, "
                    "and outgoing longwave as a stand-in for the MERSI-RM thermal-infrared window; "
                    "FY-3G carries no broadband radiometer). The MWRI-RM microwave imager is not used, as "
                    "TRMM's TMI was not.",
        "targets": "Total and convective precipitation: the maximum over the 303 km **PMR swath** (what "
                   "the PMR measures, the label), its mean over the PMR swath (the weight for "
                   "mass-capture), and for comparison the maximum over the **MERSI-RM swath** and along "
                   "the **nadir** line.",
        "enrichment": "Land/water fraction under the track, cyclic longitude, solar hour, validity flags. "
                      "No forward-view feature: MERSI-RM is a cross-track scanner.",
        "note": "The extraction refuses to save a block unless every request succeeded, and any footprint "
                "without a grid cell inside it stops the run. The range check in §2 verifies that no "
                "physically impossible values reached the dataset.",
        "label": "**\"precipitation ≥ 1 mm/hr anywhere in the PMR swath\"**, i.e. rain the scanning PMR "
                 "would measure. The swath column counts rain anywhere in the 1000 km MERSI-RM swath, "
                 "most of which lies outside the PMR's 303 km; the nadir column shows what a nadir-only "
                 "radar (like EarthCARE's CPR) would see on the same track.",
        "footprint": "in the PMR swath",
        "e4": "The PMR measures a 303 km swath inside the 1000 km MERSI-RM swath, so a policy credited "
              "with MERSI-RM-swath rain is credited with rain the radar never scans. The nadir line shows "
              "how a nadir-only radar on the same orbit would score.",
        "tiers": "| A | everything (cloud fraction, cloud-top temperature, optical depth, longwave, geometry) |\n"
                 "| A_notau | A without optical depth (what is available at night) |\n"
                 "| B | thermal only: cloud-top temperature and longwave, plus geometry |\n"
                 "| C | geometry only |",
        "night": "Optical depth (`tautot`) needs reflected sunlight, so at night MERSI-RM has only its "
                 "thermal channels.",
    },
}


def build(m):
    T = TEXT[m.name]
    sun_sync = m.name == "earthcare"
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    md(rf"""
# {m.title} trackwise dataset and baseline results

This notebook covers the **{m.title} benchmark**: {T['question']}

It has two parts:

1. **The dataset**: how it was built, integrity checks, and what the weather in it looks like.
2. **The baseline results** (experiments E1–E11): how well simple, fixed policies can decide
   when to switch the radar on under a power budget. These are the numbers any RL agent has
   to beat.

Nothing here downloads data or re-runs the extraction. The notebook reads
`data/{m.data_path.name}` and the CSVs in `results/`, and takes the orbit and time-window
settings directly from `extract_trackwise.py` and `trackwise_missions.py`, so it always matches
the pipeline that produced the data.

**Rebuild everything** (from the repository root):

```
python3 extract_trackwise.py --mission {m.name}                                # weather along the orbit, 59 blocks
python3 task_outputs/fetch_g5nr_constants.py                                   # once: land/water grid
python3 task_outputs/enrich_trackwise.py --mission {m.name} <extraction cache> # -> data/
python  -m baselines.run_baselines --mission {m.name} --mlp                    # -> results/
```
""")

    md(rf"""
## 1. How the dataset was built

| step | what happens |
|---|---|
| **orbit** | {T['orbit']} |
| **weather** | GEOS-5 Nature Run (G5NR), a 7 km global simulation, sampled every 30 minutes at 0.0625°. |
| **time alignment** | {T['alignment']} |
| **decision** | One row per minute: should the {m.radar_name} be on? 30 decisions share each 30-minute weather snapshot. |
| **passive features** | {T['features']} |
| **targets** | {T['targets']} |
| **enrichment** | {T['enrichment']} |

{T['note']}
""")

    code(rf"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

MISSION_NAME = "{m.name}"
ROOT = Path.cwd().resolve().parents[1]                 # notebook lives in missions/<mission>/
assert (ROOT / "extract_trackwise.py").exists(), f"run this notebook from missions/{{MISSION_NAME}}/ ({{ROOT}})"
sys.path.insert(0, str(ROOT))
import extract_trackwise as X                          # the pipeline's own settings
import trackwise_missions as TM

M = TM.get(MISSION_NAME)
orb = X.orbit(MISSION_NAME)
DATA = M.data_path
RES = M.folder / "results"
MM = 3600.0                                            # kg m-2 s-1 -> mm/hr
RAIN = 1.0 / MM                                        # benchmark label: >= 1 mm/hr
GEOMS = M.geometries                                   # label geometries, the radar's own first
LABEL = f"prectot_{{GEOMS[0]}}"                        # the benchmark label column
CONV = f"preccon_{{GEOMS[0]}}"
WHERE = {{"nadir": "at nadir", "radar": f"in the {{M.radar_name}} swath", "swath": "in the passive swath"}}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
plt.rcParams.update({{"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3}})

n_tle, ep0, ep1, gap = orb.check_tle_coverage(M.startdate + X.N_FIELDS * X.G5_STEP)
print(f"mission    : {{M.title}} (NORAD {{M.norad}}), passive {{M.passive.name}} {{M.passive.swath_m / 1e3:.0f}} km, "
      f"radar {{M.radar_name}} " + (f"{{M.radar.swath_m / 1e3:.0f}} km swath" if M.radar else "nadir only"))
print(f"orbit      : {{Path(M.tle_file).relative_to(ROOT)}}  ({{n_tle}} TLEs, "
      f"{{ep0:%Y-%m-%d}} .. {{ep1:%Y-%m-%d}}, largest gap {{gap}})")
print(f"orbit start: {{M.startdate:%Y-%m-%d %H:%M}} UTC  <->  G5NR step {{X.FIELD_START}}")
print(f"window     : {{X.N_FIELDS:,}} weather steps x {{X.PER_FIELD}} decisions = "
      f"{{X.N_FIELDS * X.PER_FIELD:,}} one-minute decisions "
      f"({{X.N_FIELDS * X.G5_STEP.total_seconds() / 86400:.0f}} days)")
""")

    md(r"""
## 2. Load the dataset and check it

These checks fail loudly if the file does not match the pipeline: every column in registry
order, every decision present exactly once, each row tied to the right weather step, the hash
matching the enrichment report and the baseline run, and every weather value within physical
limits.
""")

    code(r"""
rows = pd.read_parquet(DATA)
report = json.loads((DATA.parent / "enrich_report.json").read_text())
meta = json.loads((RES / "run_meta.json").read_text())
registry = re.findall(r"^  - name: (\S+)", M.registry.read_text(), flags=re.M)

N = X.N_FIELDS * X.PER_FIELD
assert list(rows.columns) == registry, "columns differ from the mission's column registry"
assert len(rows) == N, len(rows)
assert np.array_equal(rows.decision.to_numpy(), np.arange(N)), "decisions are not 0..N-1"
assert (rows.g5_step == X.FIELD_START + rows.decision // X.PER_FIELD).all()
assert report["sha256"] == meta["data_sha256"], "results were computed on a different dataset"
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == report["sha256"], "parquet changed since enrichment"

# physical limits by variable; G5NR's wettest cell is ~400 mm/hr (TRMM, Typhoon Talim's eyewall,
# 2005-08-30), well under 0.2 kg m-2 s-1 (720 mm/hr)
RANGES = [("cld", "_mean", (0, 1)), ("tau", "", (0, 1e3)), ("cldtmp", "", (150, 330)),
          ("lwtup", "", (50, 450)), ("prec", "", (0, 0.2)), ("iwp", "", (0, 20)), ("lwp", "", (0, 20))]
def limits(c):
    hit = [r for p, s, r in RANGES if c.startswith(p) and c.endswith(s)]
    return hit[-1] if hit else None
LIMITS = {c: limits(c) for c in rows.columns if limits(c) and not c.endswith("_valid")}
audit = pd.DataFrame([{"column": c, "min": rows[c].min(), "max": rows[c].max(),
                       "NaN": int(rows[c].isna().sum()),
                       "out of range": int((rows[c].notna() & ~rows[c].between(lo, hi)).sum())}
                      for c, (lo, hi) in LIMITS.items()])
assert audit["out of range"].sum() == 0
display(audit.set_index("column"))

print(f"{len(rows):,} decisions x {rows.shape[1]} columns, sha256 {report['sha256'][:16]}...")
print(f"time: {rows.time.min():%Y-%m-%d %H:%M} -> {rows.time.max():%Y-%m-%d %H:%M} UTC (orbit clock)")
print(f"locations replaced by the nadir point (poles / dateline): {report['locations_replaced_by_nadir']:,}")
nan_tmp = rows.cldtmp_mean.isna()
print(f"cldtmp_mean NaN: {int(nan_tmp.sum()):,} (G5NR reports no cloud top anywhere in the swath; "
      f"median optical depth there {rows.loc[nan_tmp, 'tautot_mean'].median():.2f}, "
      f"{int((rows.loc[nan_tmp, LABEL] >= RAIN).sum())} rainy)")
if "lwtup_fore" in rows:
    print(f"lwtup_fore NaN: {report['lwtup_fore_nan']:,} (one per weather step: the last minute "
          "before the snapshot changes)")
""")

    md(rf"""
## 3. How often does it rain?

The benchmark label is {T['label']}
""")

    code(r"""
TARGETS = [f"{v}_{g}" for v in ("prectot", "preccon") for g in GEOMS]
THRESH = [0.1, 0.5, 1, 5, 10]
balance = pd.DataFrame({c: [100 * (rows[c] >= t / MM).mean() for t in THRESH] for c in TARGETS},
                       index=[f">= {t:g} mm/hr" for t in THRESH]).round(1)
display(balance.rename(columns=lambda c: c + " (%)"))
lab, sw = balance.loc[">= 1 mm/hr", LABEL], balance.loc[">= 1 mm/hr", "prectot_swath"]
print(f"At the 1 mm/hr label, {lab:.1f}% of minutes have rain {WHERE[GEOMS[0]]} and "
      f"{sw:.1f}% somewhere in the passive swath: the swath label would credit the radar with "
      f"{sw / lab:.1f}x more rain than it can measure.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for g in GEOMS:
    c = f"prectot_{g}"
    v = rows[c].clip(lower=1e-6 / MM) * MM
    ax[0].hist(np.log10(v), bins=70, alpha=.5, label=g)
    th = np.logspace(-2, 1.3, 30)
    ax[1].plot(th, [100 * (rows[c] >= t / MM).mean() for t in th], label=g)
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].set(xlabel="log10 precipitation (mm/hr)", ylabel="decisions", title="Precipitation distribution")
ax[1].axvline(1, color="k", ls="--", lw=1)
ax[1].set(xscale="log", xlabel="threshold (mm/hr)", ylabel="% of minutes at or above",
          title="Share of rainy minutes vs threshold")
for a in ax: a.legend()
plt.tight_layout(); plt.show()
""")

    md(r"""
## 4. Where and when it rains along the orbit
""")

    code(r"""
import cartopy.crs as ccrs

rows["rain"] = rows[LABEL] >= RAIN
lat_lim = min(90, 5 * np.ceil(rows.lat.abs().max() / 5) + 5)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2 if lat_lim == 90 else 3.2),
                       subplot_kw={"projection": ccrs.PlateCarree()})
for a, (col, title, red) in zip(ax, [
        ("rain", f"Share of minutes with rain >= 1 mm/hr ({GEOMS[0]})", np.mean),
        ("tautot_mean", f"Mean cloud optical depth ({M.passive.name}-like, daytime)", np.nanmean)]):
    d = rows if col == "rain" else rows[rows.tautot_mean > 0]
    hb = a.hexbin(d.lon, d.lat, C=d[col].astype(float), gridsize=(90, 30), cmap="turbo",
                  reduce_C_function=red, mincnt=20, transform=ccrs.PlateCarree())
    a.coastlines(linewidth=0.5); a.set_extent([-180, 180, -lat_lim, lat_lim], crs=ccrs.PlateCarree())
    a.set_title(title)
    plt.colorbar(hb, ax=a, shrink=0.75)
plt.tight_layout(); plt.show()
""")

    if sun_sync:
        code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
lat = rows.groupby(rows.lat // 5 * 5).rain.mean() * 100
ax[0].bar(lat.index, lat.values, width=4.5, color="#2b8cbe")
ax[0].set(xlabel="latitude (5° bands)", ylabel="% rainy minutes", title="Rain by latitude")
sh = rows.assign(alat=rows.lat.abs()).groupby(rows.solar_hour.round().mod(24)).agg(
    n=("rain", "size"), alat=("alat", "mean"))
ax[1].bar(sh.index, 100 * sh.n / len(rows), color="#de2d26", alpha=.8)
ax[1].set(xlabel="local solar hour", ylabel="% of all minutes", xticks=range(0, 24, 3),
          title=f"When {M.title} samples (bars) and where (line)")
ax2 = ax[1].twinx()
ax2.plot(sh.index, sh.alat, "k.-"); ax2.set(ylabel="mean |latitude| (deg)", ylim=(0, 90)); ax2.grid(False)
plt.tight_layout(); plt.show()
busy = sh.n.nlargest(6)
print(f"Sun-synchronous orbit: {100 * busy.sum() / len(rows):.0f}% of minutes fall in the 6 hours around "
      f"{sh.n.idxmax():.0f}:00 and {(sh.n.idxmax() + 12) % 24:.0f}:00 local time (mean |lat| "
      f"{sh.loc[sh.n.idxmax(), 'alat']:.0f} deg at the peak). The other hours are only reached near the "
      f"poles (mean |lat| up to {sh.alat.max():.0f} deg), so in this dataset time of day and latitude "
      f"cannot be separated: the benchmark covers {M.title}'s ~14:00/02:00 sampling only.")
""")
    else:
        code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
lat = rows.groupby(rows.lat // 5 * 5).rain.mean() * 100
ax[0].bar(lat.index, lat.values, width=4.5, color="#2b8cbe")
ax[0].set(xlabel="latitude (5° bands)", ylabel="% rainy minutes", title="Rain by latitude")
sh = rows.assign(alat=rows.lat.abs()).groupby(rows.solar_hour.round().mod(24)).agg(
    n=("rain", "size"), rain=("rain", "mean"), alat=("alat", "mean"))
ax[1].bar(sh.index, 100 * sh.n / len(rows), color="0.6")
ax[1].set(xlabel="local solar hour", ylabel="% of all minutes", xticks=range(0, 24, 3),
          title=f"When {M.title} samples")
land = rows.land_frac_nadir > 0.95
for name, sel in [("all", slice(None)), ("land", land), ("ocean", rows.ocean_frac_nadir > 0.95)]:
    d = rows[sel]
    ax[2].plot(*zip(*d.groupby(d.solar_hour.round().mod(24)).rain.mean().mul(100).items()),
               marker=".", label=name)
ax[2].set(xlabel="local solar hour", ylabel="% rainy minutes", xticks=range(0, 24, 3),
          title="Diurnal cycle of rain in the radar footprint")
ax[2].legend()
plt.tight_layout(); plt.show()
print(f"Precessing orbit: every local hour gets {100 * sh.n.min() / len(rows):.1f}-"
      f"{100 * sh.n.max() / len(rows):.1f}% of the minutes, at mean |lat| "
      f"{sh.alat.min():.0f}-{sh.alat.max():.0f} deg, so time of day and latitude are not tied together "
      "and the diurnal cycle of rain is visible directly.")
""")

    md(r"""
## 5. Dataset structure checks

* **Land/water** under the nadir track, averaged over the minute's 60 one-second points.
* **Cyclic longitude**: `sin_lon`/`cos_lon` put −180° and +180° next to each other.
* **Persistence**: how quickly rain and clouds change from one minute to the next. This
  matters for an RL agent: if consecutive minutes were near-identical, "do what you did
  last minute" would be a strong policy.
""")

    code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(rows.land_frac_nadir, bins=40, alpha=.7, color="#8c510a", label="land")
ax[0].hist(rows.ocean_frac_nadir, bins=40, alpha=.6, color="#01665e", label="water")
ax[0].set(xlabel="fraction under nadir track", ylabel="decisions", yscale="log",
          title=f"Land/water ({100 * report['mixed_surface_share']:.0f}% of minutes mixed)")
ax[0].legend()

ax[1].scatter(rows.sin_lon.iloc[::25], rows.cos_lon.iloc[::25], s=.4, alpha=.2)
ax[1].set(aspect="equal", xlabel="sin_lon", ylabel="cos_lon", title="Cyclic longitude")

lags = [1, 2, 3, 5, 10, 15, 30, 60]
PERSIST = [LABEL, "tautot_mean", "cldtot_mean", "lwtup_mean"]
for c in PERSIST:
    ax[2].plot(lags, [rows[c].autocorr(k) for k in lags], marker="o", label=c)
ax[2].axhline(np.exp(-1), color="k", ls="--", lw=.8, label="1/e")
ax[2].set(xscale="log", xticks=lags, xlabel="lag (minutes)", ylabel="autocorrelation",
          title="Minute-to-minute persistence")
ax[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("lag-1 autocorrelation: " + ", ".join(f"{c} {rows[c].autocorr(1):.2f}" for c in PERSIST))
print("The satellite moves ~450 km per minute along track.")
""")

    md(rf"""
---
# Part 2. Baseline results (E1–E11)

## How to read the results

**The task.** For each minute in the **test period (Nov 2005 – Jan 2006)**, a policy gives a
score. The radar is switched on for the highest-scoring minutes, up to a power budget.

**Key terms**

| term | meaning |
|---|---|
| **duty cycle *d*** | share of minutes the radar is on (the power budget). 20% = on for 1 minute in 5. |
| **capture@d** | share of rainy minutes (≥ 1 mm/hr {T['footprint']}) that fall in the radar-on minutes. The main metric. Random gets *d*. |
| **mass-capture@d** | share of the total *amount* of rain in radar-on minutes (heavy rain counts more). |
| **AUC** | how well the score ranks rainy above dry minutes overall (0.5 = random, 1 = perfect). |
| **AP** | average precision; like AUC but focused on the rainy minutes (random = the rain rate). |

**The policies** (fixed rules, no learning after deployment)

| policy | what it uses |
|---|---|
| P1 random | nothing: switches on at random |
| P2 geometry LR | logistic regression on where/when only (latitude, longitude, solar hour, land/water) |
| P3 latitude band | the training rain rate of the current 5° latitude band |
| P4 single threshold | one passive feature: optical depth by day, cloud-top temperature by night |
| P5 logistic regression | linear model on all passive features + geometry (Tier A) |
| P6 gradient boosting | LightGBM on the same features: non-linear, the strongest baseline |
| P7 MLP | small neural network (two hidden layers of 64) on the same features |
| P8 oracle | knows the true rain: the upper bound, not achievable |

**How the numbers are made honest.** Models are trained on Jun–Sep 2005, the decision
threshold for each budget is set on Oct 2005, and everything is scored on Nov 2005 – Jan 2006.
One weather step is dropped at each boundary so no 30-minute snapshot appears on both sides.
""")

    code(r"""
r = lambda name: pd.read_csv(RES / f"{name}.csv")
pct = lambda x: f"{100 * x:.1f}%"
print(f"label: {meta['label']}  |  split: train {meta['split']['train'][0]}..{meta['split']['train'][-1]}, "
      f"val {meta['split']['val'][0]}, test {meta['split']['test'][0]}..{meta['split']['test'][-1]}")
print(f"results computed on sha256 {meta['data_sha256'][:16]}... with lightgbm {meta['lightgbm']}, "
      f"sklearn {meta['sklearn']}")
""")

    md(r"""
## E3. Main result: how much rain does each policy catch?

The curve shows capture as the duty cycle grows. A good policy rises steeply on the left:
it catches a lot of rain while the radar is on only a small share of the time.
""")

    code(r"""
e3, curves, boot = r("e3_capture_table"), r("e3_capture_curves"), r("e10_bootstrap")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for name, g in curves.groupby("policy", sort=False):
    ref = name in ("P1 random", "P8 oracle")
    ax[0].plot(100 * g.d, 100 * g.capture, "--" if ref else "-", lw=1.6,
               color="0.55" if ref else None, label=name)
ax[0].axvline(20, color="k", lw=.8, ls=":")
ax[0].set(xlabel="duty cycle d (% of minutes radar on)", ylabel="capture (% of rainy minutes)",
          xlim=(0, 100), ylim=(0, 101), title="Capture vs duty cycle (test period)")
ax[0].legend(fontsize=8)

b = boot[boot.metric == "capture@20%"].set_index("policy")
bar = e3.set_index("policy")["capture@20%"]
colors = ["0.6" if p in ("P1 random", "P8 oracle") else "#3182bd" for p in bar.index]
ax[1].barh(bar.index, 100 * bar.values, color=colors)
for p in b.index:
    ax[1].errorbar(100 * b.loc[p, "mean"], p, xerr=[[100 * (b.loc[p, "mean"] - b.loc[p, "ci_low"])],
                   [100 * (b.loc[p, "ci_high"] - b.loc[p, "mean"])]], color="k", capsize=3)
for i, v in enumerate(bar.values):
    ax[1].text(100 * v + 1, i, f"{100 * v:.1f}%", va="center", fontsize=9)
ax[1].invert_yaxis(); ax[1].set_xlim(0, 112)
ax[1].set(xlabel="capture at 20% duty (%), with 95% bootstrap CI", title="Rain caught with the radar on 20% of the time")
plt.tight_layout(); plt.show()

show = e3.set_index("policy")[["auc", "ap", "capture@10%", "capture@20%", "capture@30%", "mass_capture@20%"]]
display(show.style.format({c: "{:.3f}" for c in ["auc", "ap"]} |
                          {c: "{:.1%}" for c in show.columns if "capture" in c}))

p6, p5, p4, p1 = (bar[k] for k in ["P6 gradient boosting", "P5 logistic regression", "P4 single threshold", "P1 random"])
oracle20 = bar["P8 oracle"]
display(Markdown(
    f"**Takeaway.** With the radar on 20% of the time, the best fixed policy (gradient boosting) "
    f"catches **{pct(p6)}** of rainy minutes, versus {pct(p1)} for random, {pct(p4)} for a "
    f"single-feature rule and {pct(p5)} for logistic regression. That is "
    f"**{p6 / p1:.1f}x random**. The oracle shows the ceiling: at 20% duty it catches {pct(oracle20)}, "
    f"since {pct(curves.base_rate.iloc[0])} of test minutes are rainy. "
    f"The small neural network (P7) reaches {pct(bar['P7 MLP'])} with AUC "
    f"{e3.set_index('policy').auc['P7 MLP']:.3f}, against gradient boosting's "
    f"{e3.set_index('policy').auc['P6 gradient boosting']:.3f} (E10 tests whether the gap is real)."))
""")

    md(r"""
## E10. Are the differences real? (bootstrap confidence intervals)

The test period is resampled 1,000 times **by whole days** (minutes within a day are not
independent). Non-overlapping intervals mean the ranking is not a fluke of which days were
in the test set.

Every policy is scored on the **same** resampled days, so we can also take the difference
between two policies on each resample (a *paired* comparison). That is the right test when two
policies are close: their separate intervals can overlap even when one is ahead on almost
every resample.
""")

    code(r"""
ci = boot.pivot(index="policy", columns="metric", values=["mean", "ci_low", "ci_high"])
tbl = pd.DataFrame({m: [f"{ci.loc[p, ('mean', m)]:.3f}  [{ci.loc[p, ('ci_low', m)]:.3f}, {ci.loc[p, ('ci_high', m)]:.3f}]"
                        for p in ci.index] for m in ["auc", "ap", "capture@13%", "capture@20%"]}, index=ci.index)
display(tbl)
order = ["P2 geometry LR", "P4 single threshold", "P5 logistic regression", "P6 gradient boosting"]
overlap = [(a, b_) for a, b_ in zip(order, order[1:])
           if ci.loc[a, ("ci_high", "capture@20%")] >= ci.loc[b_, ("ci_low", "capture@20%")]]
display(Markdown("**Ladder.** " + ("Every step up the ladder (geometry → single feature → linear → "
        "boosting) has non-overlapping 95% intervals for capture@20%: each step is a real improvement."
        if not overlap else f"Overlapping intervals: {overlap}.")))

paired = r("e10_bootstrap_paired")
best = paired.best.iloc[0]
pt = paired.assign(**{"difference [95% CI]": [f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]" for m, lo, hi in
                                              zip(paired.mean_diff, paired.ci_low, paired.ci_high)]})
display(Markdown(f"**Paired differences: {best} minus each other policy**"))
display(pt.pivot(index="other", columns="metric", values="difference [95% CI]")[["auc", "ap", "capture@13%", "capture@20%"]])
runner = paired.loc[paired.metric == "auc"].sort_values("mean_diff").other.iloc[0]
close = paired[paired.other == runner].set_index("metric")
real = (close.ci_low > 0) | (close.ci_high < 0)
display(Markdown(
    f"**Takeaway ({best} vs {runner}, the closest policy).** AUC difference {close.loc['auc', 'mean_diff']:+.3f} "
    f"[{close.loc['auc', 'ci_low']:+.3f}, {close.loc['auc', 'ci_high']:+.3f}], {best.split()[0]} ahead in "
    f"{100 * close.loc['auc', 'share_best_ahead']:.1f}% of resamples; capture@20% difference "
    f"{100 * close.loc['capture@20%', 'mean_diff']:+.1f} points "
    f"[{100 * close.loc['capture@20%', 'ci_low']:+.1f}, {100 * close.loc['capture@20%', 'ci_high']:+.1f}]. "
    + (f"The interval excludes zero for {', '.join(real[real].index)}."
       if real.any() else "Every interval includes zero: the two are statistically tied.")
    + (f" Not significant for {', '.join(real[~real].index)}." if real.any() and (~real).any() else "")))
""")

    md(r"""
## E1. Why the split matters: random rows vs a time-blocked split

A random split puts neighbouring minutes, which share the same 30-minute weather snapshot,
on both sides of train/test. That leaks information and can flatter the model.
""")

    code(r"""
e1 = r("e1_split").set_index("split")
display(e1[["n_train", "n_test", "duty", "capture", "mass_capture", "precision", "lift", "auc", "ap"]].round(3))
rb, tb = e1.loc["random 30%"], e1.loc["time-blocked"]
display(Markdown(
    f"**Takeaway.** Logistic regression at 13% duty: AUC {rb.auc:.3f} (random split) vs {tb.auc:.3f} "
    f"(time-blocked); capture {pct(rb.capture)} vs {pct(tb.capture)}. The paper reports the "
    "time-blocked numbers because they are the honest test of generalising to new weather."))
""")

    md(r"""
## E2. Does skill hold across months?

Each month is held out in turn, and each learned policy (P5–P7) is trained on all the others,
with a buffer at the month's edges (Feb 2006 has only one day, so it is merged into January).
The main split trains on Jun–Sep only and tests on Nov–Jan, which crosses a season. Comparing the
two on the same winter months shows how much that costs each model (dashed line = trained Jun–Sep
only).
""")

    code(r"""
e2 = r("e2_leave_one_month_out")
blocked = r("e3_capture_table").set_index("policy")
months = list(dict.fromkeys(e2.held_out))
winter = [mo for mo in months if mo >= meta["split"]["test"][0]]
fig, ax = plt.subplots(figsize=(10, 3.8))
for p, g in e2.groupby("policy", sort=False):
    line, = ax.plot(g.held_out, g.auc, marker="o", label=f"{p} (held-out month)")
    ax.hlines(blocked.loc[p, "auc"], winter[0], winter[-1], color=line.get_color(), ls="--")
ax.axvspan(months.index(winter[0]) - 0.5, months.index(winter[-1]) + 0.5, color="0.5", alpha=0.1)
ax.set(ylabel="ROC-AUC", title="Leave-one-month-out vs. trained Jun–Sep only (dashed, shaded test block)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
display(e2.pivot(index="held_out", columns="policy", values="auc").round(3))
w = e2[e2.held_out.isin(winter)].groupby("policy", sort=False)[["auc", "ap"]].mean()
season = pd.DataFrame({"AUC trained Jun–Sep": blocked.loc[w.index, "auc"],
                       "AUC trained on other 7 months": w.auc,
                       "AP trained Jun–Sep": blocked.loc[w.index, "ap"],
                       "AP trained on other 7 months": w.ap})
season["AUC gain"] = season.iloc[:, 1] - season.iloc[:, 0]
display(season.round(3))
rank = e2.pivot(index="held_out", columns="policy", values="auc")
same = (rank.rank(axis=1).nunique() == 1).all()
display(Markdown(
    f"**Takeaway.** Putting the other winter months into training raises Nov–Jan AUC by "
    + ", ".join(f"{g:+.3f} ({p.split()[0]})" for p, g in season['AUC gain'].items())
    + ". That is an upper bound on the seasonal penalty, since those folds also train on 7 months "
    "instead of 4. " + ("The ranking of the policies is the same in every held-out month, so the "
    "time-blocked numbers are conservative." if same else "The ranking changes in some months.")))
""")

    md(rf"""
## E4. Label geometry: which footprint defines "rain"?

Train on one definition of "rain" and score on another, for total precipitation and for
convective precipitation (the heavy, storm-type part). {T['e4']} **Precision** is the share of
radar-on minutes that contain rain under that label.
""")

    code(r"""
e4 = r("e4_label_geometry")
e4m = meta["e4"]
for var, name in [("prectot", "total precipitation"), ("preccon", "convective precipitation")]:
    g = e4[e4.variable == var]
    tbl = pd.DataFrame({
        m: g.pivot(index="trained_on", columns="scored_on", values=m).stack()
        for m in ["capture@20%", "precision@20%", "auc"]})
    display(Markdown(f"**{name.capitalize()}** (rows: trained on → scored on)"))
    display(tbl.style.format({"capture@20%": "{:.1%}", "precision@20%": "{:.1%}", "auc": "{:.3f}"}))

def cell(var, train, score, m):
    g = e4[(e4.variable == var) & (e4.trained_on == f"{var}_{train}") & (e4.scored_on == f"{var}_{score}")]
    return g[m].iloc[0]

pr, own = e4m["positive_rate"], GEOMS[0]
text = []
for var, name in [("prectot", "Total"), ("preccon", "Convective")]:
    text.append(
        f"{name} precipitation occurs in {pct(pr[f'{var}_{own}'])} of test minutes {WHERE[own]} but "
        f"{pct(pr[f'{var}_swath'])} {WHERE['swath']}; {pct(e4m['swath_pos_label_neg_share'][var])} of "
        f"swath-rainy minutes have none {WHERE[own]}, and scoring on the swath label moves precision from "
        f"{pct(cell(var, own, own, 'precision@20%'))} to {pct(cell(var, own, 'swath', 'precision@20%'))} "
        f"and capture from {pct(cell(var, own, own, 'capture@20%'))} to "
        f"{pct(cell(var, own, 'swath', 'capture@20%'))}.")
for g in GEOMS[2:]:
    text.append(f"A {g}-only radar on this orbit would find total precipitation in "
                f"{pct(pr[f'prectot_{g}'])} of minutes, against {pct(pr[f'prectot_{own}'])} {WHERE[own]}.")
display(Markdown("**Takeaway.** " + " ".join(text) + " Headline numbers must use the radar's own footprint."))
""")

    md(rf"""
## E5. Day vs night: what do we lose without sunlight?

{T['night']} Tiers remove features step by step:

| tier | features |
|---|---|
{T['tiers']}

"Switched" uses the tier's day model by day and its no-optical-depth version at night,
which is how a real policy would run.
""")

    code(r"""
e5, night = r("e5_tiers"), r("e5_night_curves")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(len(e5)); w = 0.27
for i, (c, lb) in enumerate([("day_capture@20%", "day only"), ("night_capture@20%", "night only"),
                             ("switched_capture@20%", "switched day/night")]):
    ax[0].bar(x + (i - 1) * w, 100 * e5[c], w, label=lb)
ax[0].set(xticks=x, xticklabels=e5.tier, ylabel="capture @ 20% duty (%)", title="Capture by feature tier")
for i in np.flatnonzero(e5["night_capture@20%"].isna()):
    ax[0].text(x[i], 2, "n/a\nat night", ha="center", fontsize=8)
ax[0].legend()
for name, g in night.groupby("tier", sort=False):
    ax[1].plot(100 * g.d, 100 * g.capture, label=name)
ax[1].plot([0, 100], [0, 100], "--", color="0.6", label="random")
ax[1].set(xlabel="duty cycle (%)", ylabel="capture (%)", title="Night only: capture vs duty cycle")
ax[1].legend(); plt.tight_layout(); plt.show()
display(e5.set_index("tier").style.format("{:.3f}", na_rep="n/a (no optical depth at night)"))
t = e5.set_index("tier")
display(Markdown(
    f"**Takeaway.** Running day/night switched, dropping optical depth keeps "
    f"**{pct(t.loc['A_notau', 'retention_vs_A'])}** of Tier A's capture, and thermal-only (B) keeps "
    f"{pct(t.loc['B', 'retention_vs_A'])}. Geometry alone (C) keeps {pct(t.loc['C', 'retention_vs_A'])}. "
    f"At night the best tiers catch {pct(t['night_capture@20%'].max())} at 20% duty, vs "
    f"{pct(t.loc['A', 'day_capture@20%'])} by day with optical depth."))
""")

    md(r"""
## E6. Where does the policy work best?

One global threshold (set for 20% duty overall) applied to each latitude band and surface type.
"Duty" shows how much of the budget each stratum ends up using.
""")

    code(r"""
e6 = r("e6_strata").set_index("stratum")
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].barh(e6.index, e6.auc, color="#756bb1"); ax[0].set(xlim=(0.5, 1), xlabel="AUC", title="Ranking skill by stratum")
ax[1].barh(e6.index, 100 * e6["capture@20%"], color="#3182bd", label="capture")
ax[1].barh(e6.index, 100 * e6.duty, color="#fd8d3c", alpha=.7, height=.4, label="duty used")
ax[1].axvline(20, color="k", ls=":", lw=.8)
ax[1].set(xlabel="%", title="Capture and duty used, global 20% threshold"); ax[1].legend()
for a in ax: a.invert_yaxis()
plt.tight_layout(); plt.show()
display(e6.round(3))
bands = e6.drop(index=["ocean", "land", "mixed"])
hi, lo = e6.auc.idxmax(), e6.auc.idxmin()
wet, dry = bands.base_rate.idxmax(), bands.base_rate.idxmin()
display(Markdown(
    f"**Takeaway.** Ranking is strongest for **{hi}** (AUC {e6.loc[hi, 'auc']:.3f}, rain in "
    f"{pct(e6.loc[hi, 'base_rate'])} of minutes) and weakest for **{lo}** ({e6.loc[lo, 'auc']:.3f}, "
    f"{pct(e6.loc[lo, 'base_rate'])}). The rainiest band, {wet}, uses {pct(e6.loc[wet, 'duty'])} duty under "
    f"the global threshold against {pct(e6.loc[dry, 'duty'])} in {dry}: a single threshold spends the "
    "budget where rain is most frequent."))
""")

    md(r"""
## E7. Does the conclusion depend on the rain threshold?
""")

    code(r"""
e7 = r("e7_label_sensitivity")
piv = e7.pivot(index="label", columns="model", values=["base_rate_test", "auc", "ap", "capture@d=base_rate"])
display(piv.round(3))
lg = e7[e7.model == "lgbm"].set_index("label")
lr = e7[e7.model == "lr"].set_index("label")
wins = int((lg.auc > lr.auc.reindex(lg.index)).sum())
heavy, conv = f"{LABEL} >= 5 mm/hr", f"{CONV} >= 1 mm/hr"
display(Markdown(
    "Here the duty cycle is set equal to each label's rain rate, so a perfect policy would catch 100%. "
    f"**Takeaway.** Gradient boosting beats logistic regression in AUC on {wins} of {len(lr)} labels "
    f"(boosting AUC {lg.auc.min():.3f}–{lg.auc.max():.3f}). The rarer labels (≥ 5 mm/hr: "
    f"{pct(lg.loc[heavy, 'base_rate_test'])} of minutes; convective ≥ 1 mm/hr: "
    f"{pct(lg.loc[conv, 'base_rate_test'])}) reach AP {lg.loc[heavy, 'ap']:.3f} and {lg.loc[conv, 'ap']:.3f}, "
    f"and a budget equal to their rarity catches {pct(lg.loc[heavy, 'capture@d=base_rate'])} and "
    f"{pct(lg.loc[conv, 'capture@d=base_rate'])} of them."))
""")

    md(r"""
## E8. Missing data and E9. Policy cost
""")

    code(r"""
display(r("e8_missing_data").fillna("").set_index("item"))
e9 = r("e9_policy_cost").set_index("policy")
display(e9)
gaps = {c: n for c, n in report["nan_counts"].items() if not c.endswith("_valid")}
display(Markdown(
    f"**Takeaway.** Invalid targets: {sum(report['invalid_targets'].values())}. Missing feature values "
    f"({', '.join(f'{c}: {n:,}' for c, n in gaps.items())}) are handled by imputation or natively by "
    f"LightGBM. On cost: logistic regression is {int(e9.loc['P5 logistic regression', 'parameters'])} "
    f"parameters (~{e9.loc['P5 logistic regression', 'serialized_bytes'] / 1e3:.0f} kB), while gradient "
    f"boosting is {int(e9.loc['P6 gradient boosting', 'parameters']):,} parameters "
    f"(~{e9.loc['P6 gradient boosting', 'serialized_bytes'] / 1e6:.1f} MB). All learned policies take "
    f"{e9.wall_ms_per_decision_python.min():.1f}–{e9.wall_ms_per_decision_python.max():.1f} ms per decision in "
    "Python on a laptop (mostly call overhead; it varies between runs), against a 60 s decision interval."))
""")

    md(r"""
## E11. Which passive features matter?

Two views. **Permutation importance**: shuffle one feature and see how much the best model's
AUC drops. **Single-feature AUC**: how well each feature ranks rain on its own.
""")

    code(r"""
e3i_auc = e3.set_index("policy").auc
perm = r("e11_permutation_importance").sort_values("auc_drop_mean")
single = r("e11_single_feature_auc").sort_values("auc")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].barh(perm.feature, perm.auc_drop_mean, xerr=perm.auc_drop_std, color="#31a354")
ax[0].set(xlabel="AUC drop when shuffled", title=f"Permutation importance ({meta['e11_best_policy']})")
ax[1].barh(single.feature, single.auc, color=["#de2d26" if i else "#3182bd" for i in single.inverted])
ax[1].axvline(0.5, color="k", ls=":", lw=.8)
ax[1].set(xlim=(0.45, 1), xlabel="AUC on its own (red = lower value means more rain)",
          title="Single-feature AUC")
plt.tight_layout(); plt.show()
top = perm.sort_values("auc_drop_mean", ascending=False).feature.tolist()
display(Markdown(
    f"**Takeaway.** The model leans most on **{top[0]}** and **{top[1]}**, then {top[2]} and {top[3]}. "
    f"The best single feature, `{single.feature.iloc[-1]}`, reaches AUC {single.auc.iloc[-1]:.3f} on its "
    f"own, against {e3i_auc['P5 logistic regression']:.3f} for logistic regression and "
    f"{e3i_auc['P6 gradient boosting']:.3f} for gradient boosting on all features."))
""")

    md(rf"""
## Cesium CZML export

Writes the orbit simulation for Cesium into `cesium/`: one CZML per 30-day window at full
one-minute detail, plus a downsampled full-run file. The satellite is re-propagated from the
same TLE set as the extraction. Ground markers are green when precipitation {T['footprint']} is
≥ 1 mm/hr ({m.radar_name} on), red when it is below, and grey when there is no valid decision. A beam is
drawn while the {m.radar_name} fires. Drag any file onto a Cesium viewer, for example
[Cesium Sandcastle](https://sandcastle.cesium.com), to play it.
""")

    code(r"""
sys.path.insert(0, str(ROOT / "task_outputs"))
import export_trackwise as E                           # same exporter as the command-line script

E.export_czml(E.add_fire(pd.read_parquet(DATA), M), M)
print(f"CZML files in {E.czml_dir(M).relative_to(ROOT)}/")
""")

    md(r"""
## Summary
""")

    code(r"""
e3i = e3.set_index("policy")
e2b = e2[e2.policy == "P6 gradient boosting"]
summary = pd.DataFrame({
    "value": [f"{len(rows):,} one-minute decisions, G5NR weather 2005-06-01 .. 2006-02-01",
              pct(curves.base_rate.iloc[0]),
              f"{pct(e3i.loc['P6 gradient boosting', 'capture@20%'])} (random {pct(e3i.loc['P1 random', 'capture@20%'])})",
              f"{e3i.loc['P6 gradient boosting', 'auc']:.3f}",
              f"{pct(e3i.loc['P6 gradient boosting', 'mass_capture@20%'])}",
              f"{pct(t.loc['A_notau', 'retention_vs_A'])} of full skill",
              f"{e2b.auc.min():.3f} – {e2b.auc.max():.3f}",
              ", ".join(top[:3])]},
    index=[M.title, f"rainy minutes in test (>= 1 mm/hr, {GEOMS[0]})", "best baseline capture @ 20% duty",
           "best baseline AUC", "best baseline rain-amount capture @ 20%",
           "without optical depth (night-capable)", "best baseline AUC range across held-out months",
           "most important features"])
display(summary)
""")

    md(rf"""
**What this means for the RL work.** These are the numbers to beat: a learned gating policy
under the same 20% budget has to catch more than the gradient-boosting baseline's share of rainy
minutes on the same test period, or show that it uses the budget better (for example, by
adapting the duty cycle to the weather it sees, which a fixed threshold cannot do).

CSV exports for visualisation are produced by `task_outputs/export_trackwise.py --mission {m.name}`
into `exports/` (not in git); the Cesium CZML files go to `cesium/` (see the export cell above).
""")

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "trackwise-py", "language": "python", "name": "trackwise-py"},
                   "language_info": {"name": "python"}}
    return nb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", required=True, choices=sorted(trackwise_missions.MISSIONS))
    m = trackwise_missions.get(ap.parse_args().mission)
    out = m.folder / f"{m.file_prefix}_trackwise.ipynb"
    nb = build(m)
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
