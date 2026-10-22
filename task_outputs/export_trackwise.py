#!/usr/bin/env python3
"""Create the EarthCARE trackwise CSV exports in missions/earthcare/exports/ and the
Cesium CZML files in missions/earthcare/cesium/."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract_trackwise as X  # noqa: E402
MISSION_DIR = ROOT / "missions" / "earthcare"
SOURCE = MISSION_DIR / "data" / "EarthCare_trackwise_decisions.parquet"
EXPORTS = MISSION_DIR / "exports"
FULL_CSV = EXPORTS / "EarthCare_trackwise_decisions_full.csv"
KEPLER_CSV = EXPORTS / "EarthCare_trackwise_decisions_kepler.csv"
CZML_DIR = MISSION_DIR / "cesium"

MM_PER_KG_M2_S = 3600.0
ALTITUDE_M = 394_000.0
CZML_DAYS = 30           # one monthly CZML per 30-day window, full 1-minute detail
FULL_TRACK_STEP = 5      # full-run CZML: orbit sample every 5 minutes
FULL_MARKERS = 20_000    # full-run CZML: ~same decision thinning as the Kepler CSV
# CZML colour/beam threshold. Display only: `fire` (median split) stays the
# balanced training label, but colouring by it makes ~50% of the globe green.
SHOW_MM_HR = 1.0

# The parquet lat/lon are footprint centroids (~40 km off nadir from the roll), so the CZML
# track is re-propagated with the extraction's own TLE-set orbit (extract_trackwise.nadir).


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def subpoint(times):
    """Sub-satellite (lat, lon) in degrees at each timestamp."""
    return X.nadir(times.dt.to_pydatetime())


def build_czml(name, track, markers, multiplier, beams):
    """Satellite packet sampled from `track`, one ground marker per row of
    `markers`, and (if `beams`) a nadir beam for every `show` marker."""
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
        no_data = pd.isna(row.show)  # nadir precip invalid -> no CPR decision
        firing = not no_data and row.show == 1
        label = "n/a" if pd.isna(row.fire) else int(row.fire)
        if no_data:
            state, color = "n/a", [150, 150, 150, 110]
        elif firing:
            state, color = "ON", [0, 190, 80, 200]
        else:
            state, color = "off", [190, 40, 50, 200]
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
                    f"<b>{begin}</b><br>CPR (precip >= {SHOW_MM_HR:g} mm/hr): {state}<br>"
                    f"training label (fire): {label}<br>"
                    f"nadir precip: {row.prectot_nadir * MM_PER_KG_M2_S:.2f} mm/hr<br>"
                    f"tautot: {row.tautot_mean:.1f} | cloud: {row.cldtot_mean:.2f}"
                ),
            }
        )
        firing_count += firing

    if beams:
        # One beam entity (satellite -> sub-satellite point) shown only during
        # runs of consecutive firing minutes, instead of a polyline per firing.
        fire = markers["show"].isin([1]).to_numpy()
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


def add_fire(rows):
    """Copy of `rows` with the balanced training label `fire` (median split of valid nadir precip)."""
    valid = rows["prectot_nadir_valid"]
    threshold = rows.loc[valid, "prectot_nadir"].quantile(0.50)
    rows = rows.copy()
    rows["fire"] = pd.NA
    rows.loc[valid, "fire"] = (rows.loc[valid, "prectot_nadir"] > threshold).astype("int8")
    return rows


def export_czml(rows):
    """Write the monthly and full-run CZML files for `rows` (with `fire`) into CZML_DIR."""
    valid = rows["prectot_nadir_valid"]
    rows = rows.copy()
    # CZML-only display column; not written to the CSVs.
    rows["show"] = pd.NA
    rows.loc[valid, "show"] = (
        rows.loc[valid, "prectot_nadir"] * MM_PER_KG_M2_S >= SHOW_MM_HR).astype("int8")
    print(f"CZML colour: nadir precip >= {SHOW_MM_HR:g} mm/hr "
          f"({100 * rows.loc[valid, 'show'].mean():.1f}% green; "
          f"training label fire is {100 * rows.loc[valid, 'fire'].mean():.1f}% positive)")

    CZML_DIR.mkdir(parents=True, exist_ok=True)
    for old in CZML_DIR.glob("EarthCare_trackwise_decisions_*.czml"):
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
        path = CZML_DIR / f"EarthCare_trackwise_decisions_month{k + 1:02d}_{first:%Y%m%d}-{last:%Y%m%d}.czml"
        packets, fires = build_czml(
            f"EarthCARE trackwise CPR gating, month {k + 1}",
            segment, segment, multiplier=300, beams=True,
        )
        mb = write_czml(path, packets)
        print(f"CZML month: {path.name} ({len(segment):,} decisions, {fires:,} firings, {mb:.1f} MB)")

    # Full run: whole record, thinned so Cesium loads it quickly.
    track = rows.iloc[::FULL_TRACK_STEP]
    markers = rows.iloc[::max(len(rows) // FULL_MARKERS, 1)]
    path = CZML_DIR / "EarthCare_trackwise_decisions_full_downsampled.czml"
    packets, fires = build_czml(
        "EarthCARE trackwise CPR gating, full run (downsampled)",
        track, markers, multiplier=3600, beams=False,
    )
    mb = write_czml(path, packets)
    print(f"CZML full:  {path.name} ({len(track):,} track samples, "
          f"{len(markers):,} markers, {fires:,} firing, {mb:.1f} MB)")


def main():
    rows = pd.read_parquet(SOURCE)
    expected = (X.N_FIELDS * X.PER_FIELD, 40)
    if rows.shape != expected or not rows["decision"].is_unique:
        raise ValueError(f"unexpected trackwise dataset: {rows.shape}")

    rows = add_fire(rows)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    rows.to_csv(FULL_CSV, index=False)
    rows.iloc[::max(len(rows) // 20_000, 1)].to_csv(KEPLER_CSV, index=False)
    print(f"full CSV:   {FULL_CSV} ({len(rows):,} rows)")
    print(f"Kepler CSV: {KEPLER_CSV} ({len(rows.iloc[::max(len(rows) // 20_000, 1)]):,} rows)")

    export_czml(rows)


if __name__ == "__main__":
    main()
