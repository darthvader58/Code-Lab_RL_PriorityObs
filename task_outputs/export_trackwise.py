#!/usr/bin/env python3
"""Create a mission's track-wise CSV exports in missions/<mission>/exports/ and the Cesium CZML
files in missions/<mission>/cesium/.

    python3 task_outputs/export_trackwise.py --mission earthcare
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "task_outputs"))
import extract_trackwise as X  # noqa: E402
import trackwise_missions  # noqa: E402
from enrich_trackwise import registry  # noqa: E402

MM_PER_KG_M2_S = 3600.0
CZML_DAYS = 30           # one monthly CZML per 30-day window, full 1-minute detail
FULL_TRACK_STEP = 5      # full-run CZML: orbit sample every 5 minutes
FULL_MARKERS = 20_000    # full-run CZML: ~same decision thinning as the Kepler CSV
# CZML colour/beam threshold. Display only: `fire` (median split) stays the
# balanced training label, but colouring by it makes ~50% of the globe green.
SHOW_MM_HR = 1.0

# The parquet lat/lon are footprint centroids (EarthCARE's sit ~40 km off nadir from the roll), so
# the CZML track is re-propagated with the extraction's own TLE-set orbit (extract_trackwise.Orbit).


def label_col(m):
    """Precipitation in the radar's own footprint: the CZML colour and the `fire` label."""
    return f"prectot_{m.geometries[0]}"


def czml_dir(m):
    return m.folder / "cesium"


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def subpoint(m, times):
    """Sub-satellite (lat, lon) in degrees at each timestamp."""
    return X.orbit(m.name).nadir(times.dt.to_pydatetime())


def build_czml(m, name, track, markers, multiplier, beams):
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
            m.altitude_m,
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
            "id": m.name,
            "name": m.title,
            "description": f"Trackwise one-minute {m.radar_name} decisions from the enriched parquet dataset.",
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
                "text": m.title,
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
        no_data = pd.isna(row.show)  # radar-footprint precip invalid -> no decision
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
                    f"<b>{begin}</b><br>{m.radar_name} (precip >= {SHOW_MM_HR:g} mm/hr): {state}<br>"
                    f"training label (fire): {label}<br>"
                    f"{m.geometries[0]} precip: {getattr(row, label_col(m)) * MM_PER_KG_M2_S:.2f} mm/hr<br>"
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
                "name": f"{m.radar_name} beam (firing)",
                "availability": [
                    f"{iso(a)}/{iso(b + pd.Timedelta(minutes=2))}" for a, b in runs.itertuples(index=False)
                ],
                "polyline": {
                    "positions": {"references": [f"{m.name}#position", "nadir#position"]},
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


def add_fire(rows, m):
    """Copy of `rows` with the balanced training label `fire` (median split of valid precipitation
    in the radar's footprint)."""
    col = label_col(m)
    valid = rows[f"{col}_valid"]
    threshold = rows.loc[valid, col].quantile(0.50)
    rows = rows.copy()
    rows["fire"] = pd.NA
    rows.loc[valid, "fire"] = (rows.loc[valid, col] > threshold).astype("int8")
    return rows


def export_czml(rows, m):
    """Write the monthly and full-run CZML files for `rows` (with `fire`) into czml_dir(m)."""
    col = label_col(m)
    out_dir = czml_dir(m)
    valid = rows[f"{col}_valid"]
    rows = rows.copy()
    # CZML-only display column; not written to the CSVs.
    rows["show"] = pd.NA
    rows.loc[valid, "show"] = (
        rows.loc[valid, col] * MM_PER_KG_M2_S >= SHOW_MM_HR).astype("int8")
    print(f"CZML colour: {m.geometries[0]} precip >= {SHOW_MM_HR:g} mm/hr "
          f"({100 * rows.loc[valid, 'show'].mean():.1f}% green; "
          f"training label fire is {100 * rows.loc[valid, 'fire'].mean():.1f}% positive)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{m.file_prefix}_trackwise_decisions_*.czml"):
        old.unlink()
    rows = rows.sort_values("time").reset_index(drop=True)
    # Satellite at each decision time; markers at mid-minute (footprint centre).
    rows["sat_lat"], rows["sat_lon"] = subpoint(m, rows["time"])
    rows["mid_lat"], rows["mid_lon"] = subpoint(m, rows["time"] + pd.Timedelta(seconds=30))
    start = rows["time"].min()

    # Monthly: consecutive 30-day windows at full 1-minute resolution.
    window = ((rows["time"] - start) // pd.Timedelta(days=CZML_DAYS)).astype(int)
    for k, segment in rows.groupby(window):
        first, last = segment["time"].min(), segment["time"].max()
        path = out_dir / f"{m.file_prefix}_trackwise_decisions_month{k + 1:02d}_{first:%Y%m%d}-{last:%Y%m%d}.czml"
        packets, fires = build_czml(
            m, f"{m.title} trackwise {m.radar_name} gating, month {k + 1}",
            segment, segment, multiplier=300, beams=True,
        )
        mb = write_czml(path, packets)
        print(f"CZML month: {path.name} ({len(segment):,} decisions, {fires:,} firings, {mb:.1f} MB)")

    # Full run: whole record, thinned so Cesium loads it quickly.
    track = rows.iloc[::FULL_TRACK_STEP]
    markers = rows.iloc[::max(len(rows) // FULL_MARKERS, 1)]
    path = out_dir / f"{m.file_prefix}_trackwise_decisions_full_downsampled.czml"
    packets, fires = build_czml(
        m, f"{m.title} trackwise {m.radar_name} gating, full run (downsampled)",
        track, markers, multiplier=3600, beams=False,
    )
    mb = write_czml(path, packets)
    print(f"CZML full:  {path.name} ({len(track):,} track samples, "
          f"{len(markers):,} markers, {fires:,} firing, {mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", required=True, choices=sorted(trackwise_missions.MISSIONS))
    m = trackwise_missions.get(ap.parse_args().mission)
    rows = pd.read_parquet(m.data_path)
    expected = (X.N_FIELDS * X.PER_FIELD, len(registry(m.registry)))
    if rows.shape != expected or not rows["decision"].is_unique:
        raise ValueError(f"unexpected track-wise dataset: {rows.shape}, need {expected}")

    rows = add_fire(rows, m)
    exports = m.folder / "exports"
    full_csv = exports / f"{m.file_prefix}_trackwise_decisions_full.csv"
    kepler_csv = exports / f"{m.file_prefix}_trackwise_decisions_kepler.csv"
    exports.mkdir(parents=True, exist_ok=True)
    rows.to_csv(full_csv, index=False)
    rows.iloc[::max(len(rows) // 20_000, 1)].to_csv(kepler_csv, index=False)
    print(f"full CSV:   {full_csv} ({len(rows):,} rows)")
    print(f"Kepler CSV: {kepler_csv} ({len(rows.iloc[::max(len(rows) // 20_000, 1)]):,} rows)")

    export_czml(rows, m)


if __name__ == "__main__":
    main()
