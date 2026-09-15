#!/usr/bin/env python3
"""Steps 2-3-4 (grid/partition validation) + 5-8 (true nadir re-propagation,
land/ocean fraction per decision, output write + report) for the land/water
nadir feature task.

Reads ONLY:
  - trackwise_decisions.parquet (repo root) -- for the set of `decision` values
  - task_outputs/const_cache/*.npy          -- cached from step1 (no network)

Writes ONLY:
  - task_outputs/t4_land_water.parquet

Never touches trackwise_decisions.parquet, never re-fetches, never opens
inst30mn_2d_met1_Nx.
"""
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HERE, "const_cache")
OUT_PARQUET = os.path.join(HERE, "t4_land_water.parquet")
PARQUET_IN = os.path.join(REPO, "trackwise_decisions.parquet")

# ---- exact constants from extract_trackwise.py (read-only reference) ------
DECISION = timedelta(minutes=1)
SUB_STEP = timedelta(seconds=1)
SUB_PER_DEC = int(DECISION / SUB_STEP)  # 60
startdate = datetime(2025, 7, 19, 15, 4, tzinfo=timezone.utc)
TLE = ["1 59908U 24101A   25200.34125573  .00010433  00000+0  14571-3 0  9999",
       "2 59908  97.0168 326.4971 0001222 108.6708 251.4681 15.57041891 64775"]

from tatc.constants import timescale
from skyfield.api import EarthSatellite, wgs84

report_lines = []
def rep(s):
    print(s, flush=True)
    report_lines.append(s)

# ---------------------------------------------------------------------------
# STEP 2: verify grid axes
# ---------------------------------------------------------------------------
rep("=== STEP 2: grid axis verification ===")
lat = np.load(os.path.join(CACHE_DIR, "lat.npy"))
lon = np.load(os.path.join(CACHE_DIR, "lon.npy"))

n_lat, n_lon = lat.size, lon.size
lat_expected = np.linspace(-90, 90, 2881)
# G5NR/GEOS lon convention: -180 to 180-0.0625, i.e. 5760 pts spanning 360 deg
lon_expected = -180.0 + 0.0625 * np.arange(5760)

lat_match = (n_lat == lat_expected.size) and np.allclose(lat, lat_expected, atol=1e-9)
lon_match_standard = (n_lon == lon_expected.size) and np.allclose(lon, lon_expected, atol=1e-9)
# also check 0..360 convention in case that's what's returned
lon_expected_0_360 = 0.0 + 0.0625 * np.arange(5760)
lon_match_0_360 = (n_lon == lon_expected_0_360.size) and np.allclose(lon, lon_expected_0_360, atol=1e-9)

lat_spacing = np.diff(lat)
lon_spacing = np.diff(lon)

rep(f"lat: n={n_lat}, min={lat.min():.6f}, max={lat.max():.6f}, "
    f"spacing min/max={lat_spacing.min():.8f}/{lat_spacing.max():.8f}")
rep(f"lon: n={n_lon}, min={lon.min():.6f}, max={lon.max():.6f}, "
    f"spacing min/max={lon_spacing.min():.8f}/{lon_spacing.max():.8f}")
rep(f"lat matches analytic np.linspace(-90,90,2881): {lat_match}")
rep(f"lon matches analytic -180..180-0.0625 (step 0.0625): {lon_match_standard}")
rep(f"lon matches analytic 0..360-0.0625 (step 0.0625) instead: {lon_match_0_360}")

if lon_match_standard:
    LON_MODE = "pm180"
    lon_axis_used = lon_expected
elif lon_match_0_360:
    LON_MODE = "0_360"
    lon_axis_used = lon_expected_0_360
else:
    LON_MODE = "actual"  # fall back to whatever was actually returned
    lon_axis_used = lon
    rep("WARNING: lon axis did not match either analytic convention exactly -- "
        "using ACTUAL fetched lon array for indexing, reporting discrepancy.")

if not lat_match:
    rep("WARNING: lat axis did not match analytic expectation exactly -- "
        "using ACTUAL fetched lat array for indexing, reporting discrepancy.")
    lat_axis_used = lat
else:
    lat_axis_used = lat

LAT0, DLAT = lat_axis_used[0], (lat_axis_used[-1]-lat_axis_used[0])/(lat_axis_used.size-1)
LON0, DLON = lon_axis_used[0], (lon_axis_used[-1]-lon_axis_used[0])/(lon_axis_used.size-1)
rep(f"Using for indexing: LAT0={LAT0}, DLAT={DLAT}, LON0={LON0}, DLON={DLON}, LON_MODE={LON_MODE}")

# ---------------------------------------------------------------------------
# STEP 3: validate surface partition
# ---------------------------------------------------------------------------
rep("")
rep("=== STEP 3: surface partition validation ===")
frocean = np.load(os.path.join(CACHE_DIR, "frocean.npy")).astype("float64")
frland = np.load(os.path.join(CACHE_DIR, "frland.npy")).astype("float64")
frlandice = np.load(os.path.join(CACHE_DIR, "frlandice.npy")).astype("float64")
frlake = np.load(os.path.join(CACHE_DIR, "frlake.npy")).astype("float64")

s = frocean + frland + frlandice + frlake
rep(f"s = frocean+frland+frlandice+frlake: min={s.min():.6f} max={s.max():.6f} mean={s.mean():.6f}")
out_of_range = (s < 0.99) | (s > 1.01)
n_bad = int(out_of_range.sum())
frac_bad = n_bad / s.size
rep(f"cells outside [0.99,1.01]: {n_bad} / {s.size} ({frac_bad*100:.4f}%)")

if n_bad > 0:
    lat2d = np.broadcast_to(lat_axis_used[:, None], s.shape)
    lon2d = np.broadcast_to(lon_axis_used[None, :], s.shape)
    bad_lat = lat2d[out_of_range]
    bad_lon = lon2d[out_of_range]
    bad_s = s[out_of_range]
    # coarse region histogram by lat band
    bands = [(-90,-60,"Antarctic <=-60"), (-60,-30,"S mid lat"), (-30,30,"tropics"),
             (30,60,"N mid lat"), (60,84,"N high lat 60-84"), (84,90.001,"Arctic >84")]
    rep("Discrepancy by lat band:")
    for lo_b, hi_b, name in bands:
        m = (bad_lat >= lo_b) & (bad_lat < hi_b)
        cnt = int(m.sum())
        if cnt:
            rep(f"  {name}: {cnt} cells, s range [{bad_s[m].min():.4f}, {bad_s[m].max():.4f}]")
    rep(f"overall bad-s range: [{bad_s.min():.6f}, {bad_s.max():.6f}]")
else:
    rep("No cells outside [0.99, 1.01].")

# ---------------------------------------------------------------------------
# STEP 4: does frland already include land ice?
# ---------------------------------------------------------------------------
rep("")
rep("=== STEP 4: does frland already include frlandice? ===")
lat2d_full = np.broadcast_to(lat_axis_used[:, None], s.shape)
# normalize lon to -180..180 for region masks regardless of LON_MODE
lon_pm180 = ((lon_axis_used + 180) % 360) - 180
lon2d_full = np.broadcast_to(lon_pm180[None, :], s.shape)

antarctica_mask = lat2d_full <= -60
greenland_mask = (lat2d_full >= 60) & (lat2d_full <= 84) & (lon2d_full >= -75) & (lon2d_full <= -10)

def region_stats(mask, name):
    a_only = (frland + frocean + frlake)[mask]
    b_with_ice = (frland + frlandice + frocean + frlake)[mask]
    rep(f"{name}: n_cells={int(mask.sum())}")
    rep(f"  (a) frland+frocean+frlake       mean={a_only.mean():.4f} (min={a_only.min():.4f}, max={a_only.max():.4f})")
    rep(f"  (b) frland+frlandice+frocean+frlake mean={b_with_ice.mean():.4f} (min={b_with_ice.min():.4f}, max={b_with_ice.max():.4f})")
    rep(f"  mean frlandice alone in region: {frlandice[mask].mean():.4f}")
    rep(f"  mean frland alone in region:    {frland[mask].mean():.4f}")
    return a_only.mean(), b_with_ice.mean()

a_ant, b_ant = region_stats(antarctica_mask, "Antarctica (lat<=-60)")
a_grn, b_grn = region_stats(greenland_mask, "Greenland (lat 60..84, lon -75..-10)")

includes_ice = (abs(a_ant - 1.0) < abs(b_ant - 1.0)) and (a_ant > 0.9)
rep(f"CONCLUSION: frland {'ALREADY INCLUDES' if includes_ice else 'EXCLUDES'} land ice "
    f"(evidence: Antarctica (a)={a_ant:.4f} vs (b)={b_ant:.4f}; "
    f"Greenland (a)={a_grn:.4f} vs (b)={b_grn:.4f}).")
rep("Comment-convention for later code: "
    f"# frland INCLUDES frlandice per G5NR const_2d_asm_Nx (verified empirically: "
    f"Antarctica/Greenland frland+frocean+frlake already ~1.0; adding frlandice again would double-count) "
    if includes_ice else
    "Comment-convention for later code: "
    f"# frland EXCLUDES frlandice per G5NR const_2d_asm_Nx (verified empirically: "
    f"frland+frocean+frlake alone falls well short of 1.0 over ice sheets; frlandice must be added separately) ")

FRLAND_INCLUDES_ICE = includes_ice

# ---------------------------------------------------------------------------
# STEP 5: true nadir re-propagation
# ---------------------------------------------------------------------------
rep("")
rep("=== STEP 5: true nadir sub-satellite re-propagation ===")
df = pd.read_parquet(PARQUET_IN, columns=["decision"])
decisions = np.sort(df["decision"].unique().astype(np.int64))
n_dec = decisions.size
rep(f"decisions present in trackwise_decisions.parquet: {n_dec}")
assert n_dec == 351639, f"expected 351639 decisions, got {n_dec}"

SF = EarthSatellite(TLE[0], TLE[1], "EarthCare", timescale)

CHUNK = 300_000  # decisions per skyfield batch call (300k*60 = 18M timestamps per chunk is a lot; use smaller)
CHUNK = 50_000   # 50k decisions * 60 = 3M timestamps per call -- reasonable batch size

off_seconds = np.arange(SUB_PER_DEC, dtype=np.float64)  # 0..59

nadir_lat_mean = np.empty(n_dec, dtype="float64")
nadir_lon_mean = np.empty(n_dec, dtype="float64")  # not strictly needed but useful for QA
land_frac = np.empty(n_dec, dtype="float64")
ocean_frac = np.empty(n_dec, dtype="float64")

# preload for nearest-neighbor indexing
n_lat_grid = lat_axis_used.size
n_lon_grid = lon_axis_used.size

total_subpoints = 0
t_start_epoch = pd.Timestamp(startdate)

for c0 in range(0, n_dec, CHUNK):
    c1 = min(c0 + CHUNK, n_dec)
    dec_chunk = decisions[c0:c1]  # shape (m,)
    m = dec_chunk.size
    # seconds since startdate for each (decision, sub-step) pair
    secs = (dec_chunk[:, None].astype(np.float64) * DECISION.total_seconds()
            + off_seconds[None, :]).ravel()
    times = t_start_epoch + pd.to_timedelta(secs, unit="s")
    ts = timescale.from_datetimes(times)
    sp = wgs84.subpoint(SF.at(ts))
    sub_lat = sp.latitude.degrees.reshape(m, SUB_PER_DEC)
    sub_lon = sp.longitude.degrees.reshape(m, SUB_PER_DEC)
    total_subpoints += sub_lat.size

    # nearest-neighbor index into cached grid
    # lat index
    lat_idx = np.rint((sub_lat - LAT0) / DLAT).astype(np.int64)
    np.clip(lat_idx, 0, n_lat_grid - 1, out=lat_idx)
    # lon: normalize sub_lon to same convention as lon_axis_used, then wrap
    if LON_MODE == "0_360":
        sub_lon_conv = np.mod(sub_lon, 360.0)
    else:
        # pm180 or actual (assume actual behaves like pm180-ish); normalize to -180..180
        sub_lon_conv = ((sub_lon + 180.0) % 360.0) - 180.0
    lon_idx = np.rint((sub_lon_conv - LON0) / DLON).astype(np.int64)
    lon_idx = np.mod(lon_idx, n_lon_grid)  # wrap

    fl = frland[lat_idx, lon_idx]
    fo = frocean[lat_idx, lon_idx]
    land_frac[c0:c1] = fl.mean(axis=1)
    ocean_frac[c0:c1] = fo.mean(axis=1)
    nadir_lat_mean[c0:c1] = sub_lat.mean(axis=1)
    nadir_lon_mean[c0:c1] = sub_lon.mean(axis=1)

    rep(f"  chunk {c0}:{c1} ({m} decisions) done, cumulative subpoints={total_subpoints}")

rep(f"Total decisions nadir-propagated: {n_dec}")
rep(f"Total sub-point evaluations: {total_subpoints} (expected {n_dec*SUB_PER_DEC})")
assert total_subpoints == n_dec * SUB_PER_DEC

# ---------------------------------------------------------------------------
# STEP 7: write output
# ---------------------------------------------------------------------------
rep("")
rep("=== STEP 7: write output parquet ===")
out_df = pd.DataFrame({
    "decision": decisions.astype(np.int64),
    "land_frac_nadir": land_frac.astype(np.float64),
    "ocean_frac_nadir": ocean_frac.astype(np.float64),
})
assert len(out_df) == 351639
assert list(out_df.columns) == ["decision", "land_frac_nadir", "ocean_frac_nadir"]
out_df.to_parquet(OUT_PARQUET, index=False)
rep(f"Wrote {OUT_PARQUET}: {len(out_df)} rows, columns={list(out_df.columns)}")

# ---------------------------------------------------------------------------
# STEP 8: validate and report
# ---------------------------------------------------------------------------
rep("")
rep("=== STEP 8: land_frac_nadir distribution & consistency check ===")
q = out_df["land_frac_nadir"].quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
rep("land_frac_nadir quantiles:")
for k, v in q.items():
    rep(f"  q{k:.2f} = {v:.4f}")

hist_edges = np.linspace(0, 1, 11)
hist_counts, _ = np.histogram(out_df["land_frac_nadir"], bins=hist_edges)
rep("coarse histogram (10 bins over [0,1]):")
for i in range(10):
    rep(f"  [{hist_edges[i]:.1f},{hist_edges[i+1]:.1f}): {hist_counts[i]} "
        f"({hist_counts[i]/n_dec*100:.2f}%)")

mean_land = out_df["land_frac_nadir"].mean()
mean_ocean = out_df["ocean_frac_nadir"].mean()
rep(f"orbit-sampled mean land_frac_nadir across all {n_dec} decisions: {mean_land:.4f}")
rep(f"orbit-sampled mean ocean_frac_nadir across all {n_dec} decisions: {mean_ocean:.4f}")
rep(f"Earth's true land fraction is ~29%. This orbit-sampled mean is "
    f"{'HIGHER' if mean_land > 0.29 else 'LOWER'} than 29% by {abs(mean_land-0.29)*100:.1f} pct points.")
rep("EarthCARE's near-polar sun-synchronous orbit (inclination ~97 deg) spends "
    "disproportionately more time at high latitudes than a uniform-area sample would, "
    "and high latitudes (Antarctica, Arctic, northern Eurasia/Canada) skew more land-heavy, "
    "so an upward bias relative to 29% is the expected direction.")
rep(f"{'This is CONSISTENT with the expected upward bias.' if mean_land > 0.29 else 'This is NOT consistent with the expected upward bias -- unexpected, investigate.'}")

# save report to a text file too, for convenience (not a tracked artifact requirement, just local aid)
with open(os.path.join(HERE, "t4_land_water_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print("\nDONE.")
