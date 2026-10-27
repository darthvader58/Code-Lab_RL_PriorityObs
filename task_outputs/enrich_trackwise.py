#!/usr/bin/env python3
"""Build a mission's enriched track-wise dataset from its 59 extraction parts.

    python3 task_outputs/fetch_g5nr_constants.py                                  # once: land/water grid
    python3 task_outputs/enrich_trackwise.py --mission earthcare ~/trackwise_run/cache_8mo_jun2005_season
    python3 task_outputs/enrich_trackwise.py --mission trmm ~/trackwise_run/cache_trmm_8mo_jun2005
    python3 task_outputs/enrich_trackwise.py --mission fy3g ~/trackwise_run/cache_fy3g_8mo_jun2005_season

Reads the extraction parts and task_outputs/const_cache/, writes
missions/<mission>/data/<Prefix>_trackwise_decisions.parquet (columns in
missions/<mission>/column_registry.yaml order) and missions/<mission>/data/enrich_report.json.
Per-second nadir points come from extract_trackwise.Orbit.nadir(), the same TLE-set propagation the
extraction used. Column definitions are documented in docs/feature_set.md (EarthCARE),
docs/TRMM_trackwise_method.md (TRMM) and docs/FY3G_trackwise_method.md (FY-3G).
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
import trackwise_missions  # noqa: E402

CONST = ROOT / "task_outputs" / "const_cache"

N_DEC = X.N_FIELDS * X.PER_FIELD                       # 352,800
N_PARTS = -(-X.N_FIELDS // X.CHECKPOINT)               # 59
MAX_OFFSET_KM = 100.0         # healthy footprint centroids sit within ~50 km of nadir
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


def nadir_features(orb, lat_ax, lon_ax, land, water):
    """Per decision: mid-minute nadir point, and land/water fraction over its 60 nadir samples."""
    off = np.arange(X.SUB_PER_DEC) * X.SUB_STEP.total_seconds()
    mid = np.empty((N_DEC, 2))
    land_f, water_f = np.empty(N_DEC), np.empty(N_DEC)
    t0 = pd.Timestamp(orb.mission.startdate)
    for c0 in range(0, N_DEC, CHUNK):
        dec = np.arange(c0, min(c0 + CHUNK, N_DEC))
        secs = (dec[:, None] * X.DECISION.total_seconds() + off[None, :]).ravel()
        la, lo = orb.nadir((t0 + pd.to_timedelta(secs, unit="s")).to_pydatetime())
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


def registry(path):
    """(name, tier) for every column, in registry order."""
    return re.findall(r"^  - name: (\S+)\n    tier: (\S+)", path.read_text(), flags=re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", required=True, choices=sorted(trackwise_missions.MISSIONS))
    ap.add_argument("cache", help="extraction cache directory with the 59 rows_*.parquet parts")
    a = ap.parse_args()
    orb = X.orbit(a.mission)
    m = orb.mission
    reg = registry(m.registry)
    names = [n for n, _ in reg]
    targets = [n for n, tier in reg if tier == "target"]
    out_path, report_path = m.data_path, m.data_path.parent / "enrich_report.json"

    df = consolidate(a.cache)
    log(f"{m.title}: consolidated {len(df):,} decisions from {N_PARTS} parts")
    lat_ax, lon_ax, land, water = load_constants()
    mid, df["land_frac_nadir"], df["ocean_frac_nadir"] = nadir_features(orb, lat_ax, lon_ax, land, water)

    # location: footprint centroid, replaced by the mid-minute nadir point where the planar centroid
    # is wrong (near the poles and where the footprint polygon is split at +/-180 deg)
    bad = great_circle_km(df["lat"], df["lon"], mid[:, 0], mid[:, 1]) > MAX_OFFSET_KM
    df.loc[bad, "lat"], df.loc[bad, "lon"] = mid[bad, 0], mid[bad, 1]
    df["sin_lon"], df["cos_lon"] = np.sin(np.radians(df["lon"])), np.cos(np.radians(df["lon"]))
    t = pd.to_datetime(df["time"])
    df["solar_hour"] = (t.dt.hour + t.dt.minute / 60 + df["lon"] / 15) % 24

    # lwtup_fore (missions with a forward-looking radiometer): next decision's swath mean, only
    # within the same G5NR weather step
    if "lwtup_fore" in names:
        same_step = df["g5_step"].shift(-1).eq(df["g5_step"])
        df["lwtup_fore"] = df["lwtup_mean"].shift(-1).where(same_step)

    # target validity: finite and not a fill sentinel; invalid values become NaN, nothing imputed
    for c in targets:
        df[f"{c}_valid"] = df[c].notna() & (df[c].abs() < SENTINEL)
        df.loc[~df[f"{c}_valid"], c] = np.nan

    out = df[names]
    assert out.shape == (N_DEC, len(names)), out.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

    report = {
        "rows": len(out), "columns": out.shape[1], "sha256": digest,
        "tle_file": str(Path(m.tle_file).relative_to(ROOT)), "orbit_start": str(m.startdate),
        "g5_steps": [int(out["g5_step"].min()), int(out["g5_step"].max())],
        "locations_replaced_by_nadir": int(bad.sum()),
    }
    if "lwtup_fore" in names:
        report["lwtup_fore_nan"] = int(out["lwtup_fore"].isna().sum())
    report.update({
        "invalid_targets": {c: int((~out[f"{c}_valid"]).sum()) for c in targets},
        "nan_counts": {c: int(n) for c, n in out.isna().sum().items() if n},
        "land_frac_mean": float(out["land_frac_nadir"].mean()),
        "water_frac_mean": float(out["ocean_frac_nadir"].mean()),
        "mixed_surface_share": float(((out["land_frac_nadir"] > 0.05)
                                      & (out["land_frac_nadir"] < 0.95)).mean()),
    })
    report.update({f"prectot_{g}_ge_1mm_hr": float((out[f"prectot_{g}"] >= 1 / 3600).mean())
                   for g in m.geometries})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report, indent=2))
    log(f"wrote {out_path.relative_to(ROOT)} ({len(out):,} x {out.shape[1]}) and "
        f"{report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
