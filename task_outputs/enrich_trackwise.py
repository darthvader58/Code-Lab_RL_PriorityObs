#!/usr/bin/env python3
"""Build the enriched EarthCARE trackwise dataset from the 59 extraction parts.

    python3 task_outputs/fetch_g5nr_constants.py                     # once: land/water grid
    python3 task_outputs/enrich_trackwise.py ~/trackwise_run/cache_8mo_jun2005_season

Reads the extraction parts and task_outputs/const_cache/, writes
missions/earthcare/data/EarthCare_trackwise_decisions.parquet (352,800 x 40, columns in
docs/column_registry.yaml order) and missions/earthcare/data/enrich_report.json.
Per-second nadir points come from extract_trackwise.nadir(), the same TLE-set propagation the
extraction used. Column definitions are documented in docs/feature_set.md.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract_trackwise as X  # noqa: E402

CONST = ROOT / "task_outputs" / "const_cache"
DATA = ROOT / "missions" / "earthcare" / "data"
OUT = DATA / "EarthCare_trackwise_decisions.parquet"
REPORT = DATA / "enrich_report.json"
REGISTRY = ROOT / "docs" / "column_registry.yaml"

N_DEC = X.N_FIELDS * X.PER_FIELD                       # 352,800
N_PARTS = -(-X.N_FIELDS // X.CHECKPOINT)               # 59
TARGETS = [f"{v}_{side}" for v in X.LAB for side in ("swath", "nadir")]
MAX_OFFSET_KM = 100.0         # healthy footprint centroids sit ~40 km from nadir (5.76 deg roll)
SENTINEL = 1e6                # float32 fill values; no target is physically near this
EARTH_RADIUS_KM = 6371.0
CHUNK = 6000                  # decisions per propagation chunk (one extraction block)


def log(m):
    print(m, flush=True)


def consolidate(cache):
    parts = sorted(Path(cache).glob("rows_*.parquet"))
    assert len(parts) == N_PARTS, f"{len(parts)} parts in {cache}, need {N_PARTS}"
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("decision").reset_index(drop=True)
    assert np.array_equal(df["decision"].to_numpy(), np.arange(N_DEC)), "decisions are not 0..N-1"
    assert (df["g5_step"] == X.FIELD_START + df["decision"] // X.PER_FIELD).all()
    return df


def load_constants():
    lat, lon = np.load(CONST / "lat.npy"), np.load(CONST / "lon.npy")
    assert np.allclose(lat, np.linspace(-90, 90, 2881), atol=1e-9), "unexpected G5NR lat axis"
    assert np.allclose(lon, -180 + 0.0625 * np.arange(5760), atol=1e-9), "unexpected G5NR lon axis"
    f = {v: np.load(CONST / f"{v}.npy").astype("float64")
         for v in ("frland", "frlandice", "frocean", "frlake")}
    # frland excludes land ice in this collection (docs/feature_set.md), so add frlandice once.
    return lat, lon, f["frland"] + f["frlandice"], f["frocean"] + f["frlake"]


def nadir_features(lat_ax, lon_ax, land, water):
    """Per decision: mid-minute nadir point, and land/water fraction over its 60 nadir samples."""
    off = np.arange(X.SUB_PER_DEC) * X.SUB_STEP.total_seconds()
    mid = np.empty((N_DEC, 2))
    land_f, water_f = np.empty(N_DEC), np.empty(N_DEC)
    t0 = pd.Timestamp(X.startdate)
    for c0 in range(0, N_DEC, CHUNK):
        dec = np.arange(c0, min(c0 + CHUNK, N_DEC))
        secs = (dec[:, None] * X.DECISION.total_seconds() + off[None, :]).ravel()
        la, lo = X.nadir((t0 + pd.to_timedelta(secs, unit="s")).to_pydatetime())
        la, lo = la.reshape(len(dec), -1), lo.reshape(len(dec), -1)
        mid[dec] = np.column_stack([la[:, X.SUB_PER_DEC // 2], lo[:, X.SUB_PER_DEC // 2]])
        i = np.rint((la - lat_ax[0]) / (lat_ax[1] - lat_ax[0])).astype(int)
        j = np.rint((((lo + 180) % 360) - 180 - lon_ax[0]) / (lon_ax[1] - lon_ax[0])).astype(int) % len(lon_ax)
        land_f[dec], water_f[dec] = land[i, j].mean(axis=1), water[i, j].mean(axis=1)
        if c0 % (10 * CHUNK) == 0:
            log(f"  nadir: decisions {c0:,}..{dec[-1]:,}")
    return mid, land_f, water_f


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_RADIUS_KM * np.arccos(np.clip(c, -1.0, 1.0))


def registry_order():
    names = re.findall(r"^  - name: (\S+)", REGISTRY.read_text(), flags=re.M)
    assert len(names) == 40, f"{len(names)} columns in {REGISTRY.name}, need 40"
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cache", help="extraction cache directory with the 59 rows_*.parquet parts")
    cache = ap.parse_args().cache

    df = consolidate(cache)
    log(f"consolidated {len(df):,} decisions from {N_PARTS} parts")
    lat_ax, lon_ax, land, water = load_constants()
    mid, df["land_frac_nadir"], df["ocean_frac_nadir"] = nadir_features(lat_ax, lon_ax, land, water)

    # location: footprint centroid, replaced by the mid-minute nadir point where the planar centroid
    # is wrong (near the poles and where the footprint polygon is split at +/-180 deg)
    bad = great_circle_km(df["lat"], df["lon"], mid[:, 0], mid[:, 1]) > MAX_OFFSET_KM
    df.loc[bad, "lat"], df.loc[bad, "lon"] = mid[bad, 0], mid[bad, 1]
    df["sin_lon"], df["cos_lon"] = np.sin(np.radians(df["lon"])), np.cos(np.radians(df["lon"]))
    t = pd.to_datetime(df["time"])
    df["solar_hour"] = (t.dt.hour + t.dt.minute / 60 + df["lon"] / 15) % 24

    # lwtup_fore: next decision's swath mean, only within the same G5NR weather step
    same_step = df["g5_step"].shift(-1).eq(df["g5_step"])
    df["lwtup_fore"] = df["lwtup_mean"].shift(-1).where(same_step)

    # target validity: finite and not a fill sentinel; invalid values become NaN, nothing imputed
    for c in TARGETS:
        df[f"{c}_valid"] = df[c].notna() & (df[c].abs() < SENTINEL)
        df.loc[~df[f"{c}_valid"], c] = np.nan

    out = df[registry_order()]
    DATA.mkdir(parents=True, exist_ok=True)
    assert out.shape == (N_DEC, 40), out.shape
    out.to_parquet(OUT, index=False)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()

    report = {
        "rows": len(out), "columns": out.shape[1], "sha256": digest,
        "tle_file": str(Path(X.TLE_FILE).relative_to(ROOT)), "orbit_start": str(X.startdate),
        "g5_steps": [int(out["g5_step"].min()), int(out["g5_step"].max())],
        "locations_replaced_by_nadir": int(bad.sum()),
        "lwtup_fore_nan": int(out["lwtup_fore"].isna().sum()),
        "invalid_targets": {c: int((~out[f"{c}_valid"]).sum()) for c in TARGETS},
        "nan_counts": {c: int(n) for c, n in out.isna().sum().items() if n},
        "land_frac_mean": float(out["land_frac_nadir"].mean()),
        "water_frac_mean": float(out["ocean_frac_nadir"].mean()),
        "mixed_surface_share": float(((out["land_frac_nadir"] > 0.05)
                                      & (out["land_frac_nadir"] < 0.95)).mean()),
        "prectot_nadir_ge_1mm_hr": float((out["prectot_nadir"] >= 1 / 3600).mean()),
        "prectot_swath_ge_1mm_hr": float((out["prectot_swath"] >= 1 / 3600).mean()),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report, indent=2))
    log(f"wrote {OUT.name} ({len(out):,} x {out.shape[1]}) and {REPORT.name}")


if __name__ == "__main__":
    main()
