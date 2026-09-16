#!/usr/bin/env python3
"""Create the current trackwise CSV and Cesium exports from the parquet file."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trackwise_decisions.parquet"
FULL_CSV = ROOT / "trackwise_decisions_full.csv"
KEPLER_CSV = ROOT / "trackwise_decisions_kepler.csv"
CZML_FILE = ROOT / "trackwise_decisions.czml"

MM_PER_KG_M2_S = 3600.0
ALTITUDE_M = 394_000.0
CZML_DAYS = 3


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


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

    start = rows["time"].min()
    stop_time = start + pd.Timedelta(days=CZML_DAYS)
    segment = rows[rows["time"] < stop_time].sort_values("time")
    epoch = iso(start)
    stop = iso(segment["time"].max())

    samples = []
    for row in segment.itertuples():
        samples.extend([
            float((row.time - start).total_seconds()),
            round(float(row.lon), 4),
            round(float(row.lat), 4),
            ALTITUDE_M,
        ])

    packets = [
        {
            "id": "document",
            "name": "EarthCARE trackwise CPR gating",
            "version": "1.0",
            "clock": {
                "interval": f"{epoch}/{stop}",
                "currentTime": epoch,
                "multiplier": 300,
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
    for row in segment.itertuples():
        firing = row.fire == 1
        begin = iso(row.time)
        end = iso(row.time + pd.Timedelta(minutes=25))
        packets.append(
            {
                "id": f"d{int(row.decision)}",
                "availability": f"{begin}/{end}",
                "position": {
                    "cartographicDegrees": [float(row.lon), float(row.lat), 0.0]
                },
                "point": {
                    "pixelSize": 7,
                    "heightReference": "CLAMP_TO_GROUND",
                    "color": {"rgba": [0, 190, 80, 230] if firing else [190, 40, 50, 130]},
                },
                "description": (
                    f"<b>{begin}</b><br>CPR: {'ON' if firing else 'off'}<br>"
                    f"nadir precip: {row.prectot_nadir * MM_PER_KG_M2_S:.2f} mm/hr<br>"
                    f"tautot: {row.tautot_mean:.1f} | cloud: {row.cldtot_mean:.2f}"
                ),
            }
        )
        if firing:
            firing_count += 1
            packets.append(
                {
                    "id": f"beam{int(row.decision)}",
                    "availability": f"{begin}/{iso(row.time + pd.Timedelta(minutes=2))}",
                    "polyline": {
                        "positions": {
                            "cartographicDegrees": [
                                float(row.lon), float(row.lat), ALTITUDE_M,
                                float(row.lon), float(row.lat), 0.0,
                            ]
                        },
                        "material": {
                            "solidColor": {"color": {"rgba": [0, 255, 140, 170]}}
                        },
                        "width": 2,
                        "arcType": "NONE",
                    },
                }
            )

    CZML_FILE.write_text(json.dumps(packets))
    print(f"full CSV:   {FULL_CSV} ({len(rows):,} rows)")
    print(f"Kepler CSV: {KEPLER_CSV} ({len(rows.iloc[::max(len(rows) // 20_000, 1)]):,} rows)")
    print(f"CZML:       {CZML_FILE} ({len(segment):,} decisions, {firing_count:,} firings)")


if __name__ == "__main__":
    main()
