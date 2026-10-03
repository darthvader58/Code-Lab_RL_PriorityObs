#!/usr/bin/env python3
"""Repair lat/lon (and the lon-derived sin_lon, cos_lon, solar_hour) in the
trackwise parquet.

extract_trackwise.py stored lat/lon as the planar centroid of each 1-minute
footprint polygon. Near the poles and wherever the footprint is split at
+/-180 deg that centroid is wrong (up to ~20,000 km off). Healthy rows sit
~40 km from nadir (the 5.76 deg roll), so rows more than MAX_OFFSET_KM from the
mid-minute sub-satellite point are replaced by that point. All other rows and
all measurement columns are left untouched. Re-running is a no-op.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from export_trackwise import subpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trackwise_decisions.parquet"
MAX_OFFSET_KM = 100.0
EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_RADIUS_KM * np.arccos(np.clip(c, -1.0, 1.0))


def main():
    rows = pd.read_parquet(SOURCE)
    lat, lon = subpoint(rows["time"] + pd.Timedelta(seconds=30))
    bad = great_circle_km(rows["lat"].to_numpy(), rows["lon"].to_numpy(), lat, lon) > MAX_OFFSET_KM
    print(f"rows > {MAX_OFFSET_KM:.0f} km from nadir: {bad.sum():,} of {len(rows):,}")
    if not bad.any():
        return

    utc_hour = rows["time"].dt.hour + rows["time"].dt.minute / 60
    rows.loc[bad, "lat"] = lat[bad]
    rows.loc[bad, "lon"] = lon[bad]
    rows.loc[bad, "sin_lon"] = np.sin(np.radians(lon[bad]))
    rows.loc[bad, "cos_lon"] = np.cos(np.radians(lon[bad]))
    rows.loc[bad, "solar_hour"] = (utc_hour[bad] + lon[bad] / 15) % 24
    rows.to_parquet(SOURCE, index=False)
    print(f"patched lat, lon, sin_lon, cos_lon, solar_hour in {SOURCE.name}")


if __name__ == "__main__":
    main()
