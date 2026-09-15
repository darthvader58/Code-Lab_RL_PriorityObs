#!/usr/bin/env python3
"""Finish Task 4 from the already-fetched constants and nadir cache.

No network, weather data, extraction, or source-parquet writes occur here.
The cached arrays contain all 60 offline-propagated nadir samples per decision.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task_outputs"
C = OUT / "const_cache"
N = OUT / "nadir_cache"
src = pd.read_parquet(ROOT / "trackwise_decisions.parquet", columns=["decision"])
dec = np.load(N / "decisions.npy")
sub_lat = np.load(N / "sub_lat.npy", mmap_mode="r")
sub_lon = np.load(N / "sub_lon.npy", mmap_mode="r")
lat = np.load(C / "lat.npy")
lon = np.load(C / "lon.npy")
frland = np.load(C / "frland.npy", mmap_mode="r").astype("float64")
frlandice = np.load(C / "frlandice.npy", mmap_mode="r").astype("float64")
frocean = np.load(C / "frocean.npy", mmap_mode="r").astype("float64")
frlake = np.load(C / "frlake.npy", mmap_mode="r").astype("float64")

assert len(src) == len(dec) == 351639
assert src["decision"].is_unique and len(np.unique(dec)) == len(dec)
assert np.array_equal(src["decision"].to_numpy(), dec), "cache decisions do not match source order"
assert sub_lat.shape == sub_lon.shape == (len(dec), 60)

# The four fractions form an exclusive partition to within float32 rounding.
partition = frland + frlandice + frocean + frlake
partition_error = partition - 1.0
bad_partition = np.abs(partition_error) > 0.01

# Region checks establish from the data that frland excludes land ice: over
# Antarctica, frland+frocean+frlake is ~0.38 while adding frlandice is ~1.
L = lat[:, None]
Lo = lon[None, :]
ant = np.broadcast_to(L <= -60, partition.shape)
green = np.broadcast_to((L >= 60) & (L <= 84) & (Lo >= -75) & (Lo <= -10), partition.shape)
def region(mask):
    return {"frland_plus_water_mean": float((frland + frocean + frlake)[mask].mean()),
            "all_four_mean": float(partition[mask].mean()),
            "frland_mean": float(frland[mask].mean()),
            "frlandice_mean": float(frlandice[mask].mean())}
region_stats = {"antarctica_lat_le_-60": region(ant), "greenland": region(green)}
frland_includes_ice = False

# Nearest grid cell for every cached one-second nadir sample. The axes are
# exactly regular and the longitude axis is -180..180, so wrapping is explicit.
lat_idx = np.rint((np.asarray(sub_lat, dtype="float64") - lat[0]) / (lat[1] - lat[0])).astype(np.int64)
np.clip(lat_idx, 0, len(lat) - 1, out=lat_idx)
wrapped_lon = ((np.asarray(sub_lon, dtype="float64") + 180.0) % 360.0) - 180.0
lon_idx = np.rint((wrapped_lon - lon[0]) / (lon[1] - lon[0])).astype(np.int64)
lon_idx %= len(lon)

# frland excludes ice in this collection, so include frlandice exactly once.
land_samples = frland[lat_idx, lon_idx] + frlandice[lat_idx, lon_idx]
water_samples = frocean[lat_idx, lon_idx] + frlake[lat_idx, lon_idx]
land_frac = land_samples.mean(axis=1)
ocean_frac = water_samples.mean(axis=1)

out = pd.DataFrame({"decision": dec.astype("int64"),
                    "land_frac_nadir": land_frac,
                    "ocean_frac_nadir": ocean_frac})
assert out["decision"].is_unique
assert ((out[["land_frac_nadir", "ocean_frac_nadir"]] >= 0).all().all() and
        (out[["land_frac_nadir", "ocean_frac_nadir"]] <= 1).all().all())
out.to_parquet(OUT / "t4_land_water.parquet", index=False)

q = out["land_frac_nadir"].quantile([0, .01, .05, .10, .25, .5, .75, .9, .95, .99, 1]).to_dict()
hist, edges = np.histogram(out["land_frac_nadir"], bins=np.linspace(0, 1, 11))
report = {
    "rows": int(len(out)), "subpoints_per_decision": 60,
    "decision_alignment_exact": True, "duplicate_decisions": 0,
    "axes_match_expected_grid": bool(np.allclose(lat, np.linspace(-90, 90, 2881)) and np.allclose(lon, -180 + .0625*np.arange(5760))),
    "partition_min": float(partition.min()), "partition_max": float(partition.max()),
    "partition_mean": float(partition.mean()), "partition_max_abs_error": float(np.abs(partition_error).max()),
    "partition_exceptions_abs_gt_0.01": int(bad_partition.sum()),
    "frland_includes_ice": frland_includes_ice, "region_stats": region_stats,
    "land_definition": "frland + frlandice (frland empirically excludes land ice)",
    "water_definition": "frocean + frlake",
    "land_frac_quantiles": {str(k): float(v) for k, v in q.items()},
    "land_frac_histogram_counts_10_bins": hist.astype(int).tolist(),
    "land_frac_histogram_edges": edges.tolist(),
    "land_mean": float(out.land_frac_nadir.mean()), "ocean_mean": float(out.ocean_frac_nadir.mean()),
    "mean_sum": float((out.land_frac_nadir + out.ocean_frac_nadir).mean()),
    "mixed_surface_fraction_0.05_to_0.95": float(((out.land_frac_nadir > .05) & (out.land_frac_nadir < .95)).mean()),
    "constant_cache_marker_present": (C / "_DONE").exists(),
    "nadir_cache_marker_present": (N / "_DONE").exists(),
}
(OUT / "t4_land_water_report.json").write_text(json.dumps(report, indent=2) + "\n")
md = ["# Task 4 land/water validation", "", json.dumps(report, indent=2), "",
      "The constants partition shows that `frland` excludes land ice; land is therefore `frland + frlandice`, and water is `frocean + frlake`. The 60 cached one-second nadir samples per decision were used."]
(OUT / "t4_land_water_report.md").write_text("\n".join(md) + "\n")
print(json.dumps(report, indent=2))
