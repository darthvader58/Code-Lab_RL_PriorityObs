#!/usr/bin/env python3
"""Create the current trackwise CSV and Cesium exports from the parquet file."""
import json
from pathlib import Path

import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trackwise_decisions.parquet"
FULL_CSV = ROOT / "trackwise_decisions_full.csv"
KEPLER_CSV = ROOT / "trackwise_decisions_kepler.csv"
CZML_DIR = ROOT / "trackwise_czml"

MM_PER_KG_M2_S = 3600.0
ALTITUDE_M = 394_000.0
CZML_DAYS = 30           # one monthly CZML per 30-day window, full 1-minute detail
FULL_TRACK_STEP = 5      # full-run CZML: orbit sample every 5 minutes
FULL_MARKERS = 20_000    # full-run CZML: ~same decision thinning as the Kepler CSV

# Same TLE as extract_trackwise.py. The parquet lat/lon are footprint centroids
# (~40 km off nadir from the roll; broken ones repaired by
# fix_trackwise_locations.py), so the CZML track is re-propagated from the TLE.
TLE = ["1 59908U 24101A   25200.34125573  .00010433  00000+0  14571-3 0  9999",
       "2 59908  97.0168 326.4971 0001222 108.6708 251.4681 15.57041891 64775"]


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def subpoint(times):
    """Sub-satellite (lat, lon) in degrees at each timestamp."""
    ts = load.timescale(builtin=True)
    sat = EarthSatellite(TLE[0], TLE[1], "EarthCare", ts)
    sp = wgs84.subpoint(sat.at(ts.from_datetimes(times.dt.to_pydatetime())))
    return sp.latitude.degrees, sp.longitude.degrees


def build_czml(name, track, markers, multiplier, beams):
    """Satellite packet sampled from `track`, one ground marker per row of
    `markers`, and (if `beams`) a nadir beam for every firing marker."""
    start, end = track["time"].min(), track["time"].max()
    epoch, stop = iso(start), iso(end)

    samples = []
    for row in track.itertuples():
        samples.extend([
            float((row.time - start).total_seconds()),
            round(float(row.sat_lon), 4),
            round(float(row.sat_lat), 4),
            ALTITUDE_M,
        ])

    packets = [
        {
            "id": "document",
            "name": name,
            "version": "1.0",
            "clock": {
                "interval": f"{epoch}/{stop}",
                "currentTime": epoch,
                "multiplier": multiplier,
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER",
            },
        },
        {
            "id": "earthcare",
            "name": "EarthCARE",
            "description": "Trackwise one-minute CPR decisions from the enriched parquet dataset.",
            "availability": f"{epoch}/{stop}",
            "position": {
                "epoch": epoch,
                "cartographicDegrees": samples,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
            },
            "point": {
                "color": {"rgba": [255, 255, 255, 255]},
                "pixelSize": 12,
                "outlineColor": {"rgba": [40, 40, 40, 255]},
                "outlineWidth": 2,
            },
            "label": {
                "text": "EarthCARE",
                "fillColor": {"rgba": [255, 255, 255, 220]},
                "font": "12pt sans-serif",
                "pixelOffset": {"cartesian2": [14, 0]},
                "horizontalOrigin": "LEFT",
            },
            "path": {
                "material": {
                    "polylineOutline": {
                        "color": {"rgba": [120, 190, 255, 200]},
                        "outlineColor": {"rgba": [0, 0, 0, 90]},
                        "outlineWidth": 1,
                    }
                },
                "width": 3,
                "leadTime": 0,
                "trailTime": 5400,
                "resolution": 60,
            },
        },
    ]

    firing_count = 0
    for row in markers.itertuples():
        no_data = pd.isna(row.fire)  # nadir precip invalid -> no CPR decision
        firing = not no_data and row.fire == 1
        if no_data:
            state, color = "n/a", [150, 150, 150, 110]
        elif firing:
            state, color = "ON", [0, 190, 80, 230]
        else:
            state, color = "off", [190, 40, 50, 130]
        lon, lat = round(float(row.mid_lon), 4), round(float(row.mid_lat), 4)
        begin = iso(row.time)
        end = iso(row.time + pd.Timedelta(minutes=25))
        packets.append(
            {
                "id": f"d{int(row.decision)}",
                "availability": f"{begin}/{end}",
                "position": {"cartographicDegrees": [lon, lat, 0.0]},
                "point": {
                    "pixelSize": 7,
                    "heightReference": "CLAMP_TO_GROUND",
                    "color": {"rgba": color},
                },
                "description": (
                    f"<b>{begin}</b><br>CPR: {state}<br>"
                    f"nadir precip: {row.prectot_nadir * MM_PER_KG_M2_S:.2f} mm/hr<br>"
                    f"tautot: {row.tautot_mean:.1f} | cloud: {row.cldtot_mean:.2f}"
                ),
            }
        )
        firing_count += firing

    if beams:
        # One beam entity (satellite -> sub-satellite point) shown only during
        # runs of consecutive firing minutes, instead of a polyline per firing.
        fire = markers["fire"].isin([1]).to_numpy()
        gap = markers["time"].diff() != pd.Timedelta(minutes=1)
        run_id = (pd.Series(fire != pd.Series(fire).shift().to_numpy()) | gap.to_numpy()).cumsum()
        runs = markers[fire].groupby(run_id[fire].to_numpy())["time"].agg(["min", "max"])
        ground = [v if i % 4 != 3 else 0.0 for i, v in enumerate(samples)]
        packets += [
            {
                "id": "nadir",
                "position": {
                    "epoch": epoch,
                    "cartographicDegrees": ground,
                    "interpolationAlgorithm": "LAGRANGE",
                    "interpolationDegree": 5,
                },
            },
            {
                "id": "beam",
                "name": "CPR beam (firing)",
                "availability": [
                    f"{iso(a)}/{iso(b + pd.Timedelta(minutes=2))}" for a, b in runs.itertuples(index=False)
                ],
                "polyline": {
                    "positions": {"references": ["earthcare#position", "nadir#position"]},
                    "material": {"solidColor": {"color": {"rgba": [0, 255, 140, 170]}}},
                    "width": 2,
                    "arcType": "NONE",
                },
            },
        ]
    return packets, firing_count


def write_czml(path, packets):
    path.write_text(json.dumps(packets, separators=(",", ":")))
    return path.stat().st_size / 1e6


def main():
    rows = pd.read_parquet(SOURCE)
    expected = (351_639, 40)
    if rows.shape != expected or not rows["decision"].is_unique:
        raise ValueError(f"unexpected trackwise dataset: {rows.shape}")

    valid = rows["prectot_nadir_valid"]
    threshold = rows.loc[valid, "prectot_nadir"].quantile(0.50)
    rows = rows.copy()
    rows["fire"] = pd.NA
    rows.loc[valid, "fire"] = (rows.loc[valid, "prectot_nadir"] > threshold).astype("int8")
    rows.to_csv(FULL_CSV, index=False)
    rows.iloc[::max(len(rows) // 20_000, 1)].to_csv(KEPLER_CSV, index=False)
    print(f"full CSV:   {FULL_CSV} ({len(rows):,} rows)")
    print(f"Kepler CSV: {KEPLER_CSV} ({len(rows.iloc[::max(len(rows) // 20_000, 1)]):,} rows)")

    CZML_DIR.mkdir(exist_ok=True)
    for old in CZML_DIR.glob("trackwise_decisions_*.czml"):
        old.unlink()
    rows = rows.sort_values("time").reset_index(drop=True)
    # Satellite at each decision time; markers at mid-minute (footprint centre).
    rows["sat_lat"], rows["sat_lon"] = subpoint(rows["time"])
    rows["mid_lat"], rows["mid_lon"] = subpoint(rows["time"] + pd.Timedelta(seconds=30))
    start = rows["time"].min()

    # Monthly: consecutive 30-day windows at full 1-minute resolution.
    window = ((rows["time"] - start) // pd.Timedelta(days=CZML_DAYS)).astype(int)
    for k, segment in rows.groupby(window):
        first, last = segment["time"].min(), segment["time"].max()
        path = CZML_DIR / f"trackwise_decisions_month{k + 1:02d}_{first:%Y%m%d}-{last:%Y%m%d}.czml"
        packets, fires = build_czml(
            f"EarthCARE trackwise CPR gating, month {k + 1}",
            segment, segment, multiplier=300, beams=True,
        )
        mb = write_czml(path, packets)
        print(f"CZML month: {path.name} ({len(segment):,} decisions, {fires:,} firings, {mb:.1f} MB)")

    # Full run: whole record, thinned so Cesium loads it quickly.
    track = rows.iloc[::FULL_TRACK_STEP]
    markers = rows.iloc[::max(len(rows) // FULL_MARKERS, 1)]
    path = CZML_DIR / "trackwise_decisions_full_downsampled.czml"
    packets, fires = build_czml(
        "EarthCARE trackwise CPR gating, full run (downsampled)",
        track, markers, multiplier=3600, beams=False,
    )
    mb = write_czml(path, packets)
    print(f"CZML full:  {path.name} ({len(track):,} track samples, "
          f"{len(markers):,} markers, {fires:,} firing, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
