#!/usr/bin/env python3
"""Step 1: ONE-TIME fetch of the time-invariant G5NR const collection via OPeNDAP.
Caches area, frlake, frland, frlandice, frocean, phis, sgh, lat, lon as .npy
under task_outputs/const_cache/. Run this exactly once.
"""
import os, sys, time
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "const_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

URL = "https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/const/const_2d_asm_Nx"
VARS = ["area", "frlake", "frland", "frlandice", "frocean", "phis", "sgh"]

def main():
    marker = os.path.join(CACHE_DIR, "_DONE")
    if os.path.exists(marker):
        print("Already cached (marker present). Not re-fetching.")
        return
    print(f"[fetch] opening {URL}", flush=True)
    t0 = time.time()
    ds = xr.open_dataset(URL, decode_times=False)
    print(f"[fetch] opened in {time.time()-t0:.1f}s; dims={dict(ds.dims)}", flush=True)

    lat = ds["lat"].values.astype("float64")
    lon = ds["lon"].values.astype("float64")
    print(f"[fetch] lat: n={lat.size} min={lat.min()} max={lat.max()}")
    print(f"[fetch] lon: n={lon.size} min={lon.min()} max={lon.max()}")

    np.save(os.path.join(CACHE_DIR, "lat.npy"), lat)
    np.save(os.path.join(CACHE_DIR, "lon.npy"), lon)

    for v in VARS:
        t1 = time.time()
        da = ds[v].isel(time=0).load()
        arr = da.values.astype("float32")
        np.save(os.path.join(CACHE_DIR, f"{v}.npy"), arr)
        print(f"[fetch] {v}: shape={arr.shape} loaded+saved in {time.time()-t1:.1f}s "
              f"min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g} mean={np.nanmean(arr):.6g}",
              flush=True)

    ds.close()
    with open(marker, "w") as f:
        f.write("done\n")
    print(f"[fetch] TOTAL time {time.time()-t0:.1f}s. Cache complete at {CACHE_DIR}")

if __name__ == "__main__":
    main()
