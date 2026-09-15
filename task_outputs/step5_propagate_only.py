#!/usr/bin/env python3
"""Step 5 ONLY: true nadir sub-satellite re-propagation via skyfield, using the
exact TLE/startdate/DECISION/SUB_STEP from extract_trackwise.py. Pure offline
orbit propagation -- NO network access.

Caches per-decision, per-sub-step (60) lat/lon arrays to task_outputs/nadir_cache/
so the (~21M evaluation) propagation never needs to be redone, regardless of
when/whether the separate const-collection fetch (Step 1, network) succeeds.
"""
import os, time, gc
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NADIR_CACHE = os.path.join(HERE, "nadir_cache")
os.makedirs(NADIR_CACHE, exist_ok=True)
PARQUET_IN = os.path.join(REPO, "trackwise_decisions.parquet")

DECISION = timedelta(minutes=1)
SUB_STEP = timedelta(seconds=1)
SUB_PER_DEC = int(DECISION / SUB_STEP)  # 60
startdate = datetime(2025, 7, 19, 15, 4, tzinfo=timezone.utc)
TLE = ["1 59908U 24101A   25200.34125573  .00010433  00000+0  14571-3 0  9999",
       "2 59908  97.0168 326.4971 0001222 108.6708 251.4681 15.57041891 64775"]

from tatc.constants import timescale
from skyfield.api import EarthSatellite, wgs84

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def main():
    marker = os.path.join(NADIR_CACHE, "_DONE")
    if os.path.exists(marker):
        log("nadir cache already complete, skipping re-propagation.")
        return

    df = pd.read_parquet(PARQUET_IN, columns=["decision"])
    decisions = np.sort(df["decision"].unique().astype(np.int64))
    n_dec = decisions.size
    log(f"decisions present: {n_dec}")
    assert n_dec == 351639, f"expected 351639 decisions, got {n_dec}"
    np.save(os.path.join(NADIR_CACHE, "decisions.npy"), decisions)

    SF = EarthSatellite(TLE[0], TLE[1], "EarthCare", timescale)
    off_seconds = np.arange(SUB_PER_DEC, dtype=np.float64)
    t_start_epoch = pd.Timestamp(startdate)

    # Small chunks + memmap output: this box is under heavy external memory
    # pressure (other running apps), so keep peak RSS low and flush to disk
    # every chunk rather than accumulating everything in RAM.
    CHUNK = 5_000  # 5,000 decisions * 60 = 300,000 timestamps per skyfield call
    lat_path = os.path.join(NADIR_CACHE, "sub_lat.npy")
    lon_path = os.path.join(NADIR_CACHE, "sub_lon.npy")
    sub_lat_all = np.lib.format.open_memmap(
        lat_path, mode="w+", dtype="float32", shape=(n_dec, SUB_PER_DEC))
    sub_lon_all = np.lib.format.open_memmap(
        lon_path, mode="w+", dtype="float32", shape=(n_dec, SUB_PER_DEC))

    t0 = time.time()
    total_subpoints = 0
    for c0 in range(0, n_dec, CHUNK):
        c1 = min(c0 + CHUNK, n_dec)
        dec_chunk = decisions[c0:c1]
        m = dec_chunk.size
        secs = (dec_chunk[:, None].astype(np.float64) * DECISION.total_seconds()
                + off_seconds[None, :]).ravel()
        times = t_start_epoch + pd.to_timedelta(secs, unit="s")
        ts = timescale.from_datetimes(times)
        sp = wgs84.subpoint(SF.at(ts))
        sub_lat_all[c0:c1, :] = sp.latitude.degrees.reshape(m, SUB_PER_DEC).astype("float32")
        sub_lon_all[c0:c1, :] = sp.longitude.degrees.reshape(m, SUB_PER_DEC).astype("float32")
        total_subpoints += m * SUB_PER_DEC
        del secs, times, ts, sp
        if (c0 // CHUNK) % 10 == 0:
            sub_lat_all.flush(); sub_lon_all.flush(); gc.collect()
            el = time.time() - t0
            log(f"chunk {c0}:{c1} ({m} decisions) done, cumulative subpoints={total_subpoints}, "
                f"elapsed={el:.1f}s")

    sub_lat_all.flush(); sub_lon_all.flush()
    assert total_subpoints == n_dec * SUB_PER_DEC
    with open(marker, "w") as f:
        f.write(f"n_dec={n_dec}\nsub_per_dec={SUB_PER_DEC}\ntotal_subpoints={total_subpoints}\n")
    log(f"DONE. total subpoints={total_subpoints}, elapsed={time.time()-t0:.1f}s")
    log(f"Cached to {NADIR_CACHE}: decisions.npy, sub_lat.npy, sub_lon.npy")

if __name__ == "__main__":
    main()
