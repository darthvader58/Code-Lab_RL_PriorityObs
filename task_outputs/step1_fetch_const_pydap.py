#!/usr/bin/env python3
"""Fetch the one permitted time-invariant G5NR constants collection via pydap.

This avoids xarray's optional-backend discovery, which is broken in the local
environment. It writes the same cache expected by step5to8_nadir_landwater.py.
"""
import os
import time
import numpy as np
from pydap.client import open_url

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "const_cache")
URL = "https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/0.0625_deg/const/const_2d_asm_Nx"
VARS = ["area", "frlake", "frland", "frlandice", "frocean", "phis", "sgh"]

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    marker = os.path.join(CACHE_DIR, "_DONE")
    if os.path.exists(marker):
        print("Already cached (marker present). Not re-fetching.", flush=True)
        return
    t0 = time.time()
    print(f"[fetch] opening {URL} with pydap", flush=True)
    ds = open_url(URL)
    print(f"[fetch] variables={list(ds.keys())}", flush=True)
    lat = np.asarray(ds["lat"][:], dtype="float64")
    lon = np.asarray(ds["lon"][:], dtype="float64")
    print(f"[fetch] axes: lat={lat.shape}, lon={lon.shape}", flush=True)
    np.save(os.path.join(CACHE_DIR, "lat.npy"), lat)
    np.save(os.path.join(CACHE_DIR, "lon.npy"), lon)
    for v in VARS:
        t1 = time.time()
        arr = np.asarray(ds[v][0, :, :], dtype="float32")
        np.save(os.path.join(CACHE_DIR, f"{v}.npy"), arr)
        print(f"[fetch] {v}: shape={arr.shape} min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g} mean={np.nanmean(arr):.6g} ({time.time()-t1:.1f}s)", flush=True)
    with open(marker, "w") as f:
        f.write("done\n")
    print(f"[fetch] complete in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
