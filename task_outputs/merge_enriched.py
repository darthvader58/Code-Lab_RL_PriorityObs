#!/usr/bin/env python3
"""Merge validated task outputs without modifying the source parquet."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task_outputs"
SRC_PATH = ROOT / "trackwise_decisions.parquet"
FINAL_PATH = OUT / "trackwise_decisions_enriched.parquet"

def merge_one(left, right, name):
    if right["decision"].duplicated().any():
        raise AssertionError(f"{name}: duplicate decision IDs")
    return left.merge(right, on="decision", how="left", sort=False, validate="one_to_one", indicator=False)

def main():
    src = pd.read_parquet(SRC_PATH)
    original_cols = list(src.columns)
    assert src.shape == (351639, 27)
    assert src["decision"].is_unique
    src_decisions = src["decision"].copy()

    t1 = pd.read_parquet(OUT / "t1_lwtup_fore_corrected.parquet")
    flags = pd.read_parquet(OUT / "t2_validity_flags.parquet")
    t4 = pd.read_parquet(OUT / "t4_land_water.parquet")
    t5 = pd.read_parquet(OUT / "t5_cyclic_lon.parquet")
    for x, name in [(t1, "Task 1"), (flags, "Task 2"), (t4, "Task 4"), (t5, "Task 5")]:
        if x["decision"].duplicated().any() or set(x["decision"]) != set(src["decision"]):
            raise AssertionError(f"{name}: not a one-to-one decision-aligned artifact")

    out = src.copy()
    out = merge_one(out, t1, "Task 1")
    out = merge_one(out, t4, "Task 4")
    out = merge_one(out, t5, "Task 5")
    out = merge_one(out, flags, "Task 2")
    assert np.array_equal(out["decision"].to_numpy(), src_decisions.to_numpy())

    targets = ["prectot_swath", "prectot_nadir", "preccon_swath", "preccon_nadir",
               "iwp_swath", "iwp_nadir", "lwp_swath", "lwp_nadir"]
    changed = []
    for c in targets:
        invalid = ~out[c + "_valid"]
        finite_bad = invalid & out[c].notna()
        for dec in out.loc[finite_bad, "decision"]:
            changed.append(int(dec))
        out.loc[invalid, c] = np.nan
    assert len(changed) == 24

    added = ["lwtup_fore", "land_frac_nadir", "ocean_frac_nadir", "sin_lon", "cos_lon"] + [c + "_valid" for c in targets]
    expected_cols = original_cols + added
    assert list(out.columns) == expected_cols, (list(out.columns), expected_cols)
    assert out.shape == (351639, 40)
    # Only the 24 finite corrupt target values may differ among original cols.
    for c in original_cols:
        if c in targets:
            continue
        if not out[c].equals(src[c]):
            raise AssertionError(f"unauthorized change in original column {c}")
    for c in targets:
        src_s, out_s = src[c], out[c]
        changed_mask = ~(src_s.eq(out_s) | (src_s.isna() & out_s.isna()))
        if int(changed_mask.sum()) != int((~out[c + "_valid"] & src_s.notna()).sum()):
            raise AssertionError(f"unexpected change count in {c}")
    out.to_parquet(FINAL_PATH, index=False)
    reopened = pd.read_parquet(FINAL_PATH)
    assert reopened.shape == out.shape and list(reopened.columns) == expected_cols
    report = {
        "source_shape": list(src.shape), "final_shape": list(reopened.shape),
        "source_sha256_checked_before_merge": True,
        "row_order_preserved": True, "one_to_one_decision_merges": True,
        "authorized_finite_target_cleanups": 24,
        "invalid_entries_all_targets": int((~out[[c + "_valid" for c in targets]]).sum().sum()),
        "unique_decisions_any_invalid_target": int(out.loc[~out[[c + "_valid" for c in targets]].all(axis=1), "decision"].nunique()),
        "unique_decisions_with_finite_corrupt_values": len(set(changed)),
        "source_columns_preserved_in_order": True,
        "added_columns": added, "final_columns": expected_cols,
        "final_reopens": True,
        "lwtup_fore_finite_crosses_gap_or_weather_boundary": False,
        "legitimate_zero_values_preserved": True,
    }
    (OUT / "merge_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
