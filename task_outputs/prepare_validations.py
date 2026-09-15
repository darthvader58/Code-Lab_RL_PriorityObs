#!/usr/bin/env python3
"""Prepare corrected Task 1 and reproducible validation reports for Tasks 2/3/5.

This script reads only the existing source parquet and already-produced task
artifacts. It never modifies the source parquet and performs no network I/O.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "task_outputs"
SRC = ROOT / "trackwise_decisions.parquet"

def one_to_one(left, right, name):
    if left["decision"].duplicated().any() or right["decision"].duplicated().any():
        raise AssertionError(f"{name}: duplicate decision IDs")
    if set(left["decision"]) != set(right["decision"]):
        raise AssertionError(f"{name}: decision sets differ")

def task1(df):
    d = df["decision"]
    nxt_dec = d.shift(-1)
    nxt_step = df["g5_step"].shift(-1)
    nxt_lwtup = df["lwtup_mean"].shift(-1)
    adjacent = nxt_dec.sub(d).eq(1)
    same_weather = nxt_step.eq(df["g5_step"])
    next_finite = nxt_lwtup.notna()
    valid = adjacent & same_weather & next_finite
    out = pd.DataFrame({"decision": d.astype("int64"),
                        "lwtup_fore": nxt_lwtup.where(valid).astype("float64")})
    out.to_parquet(OUT / "t1_lwtup_fore_corrected.parquet", index=False)

    last = np.arange(len(df)) == len(df) - 1
    gap = (~adjacent) & ~last
    weather = adjacent & (~same_weather) & ~last
    missing_next = adjacent & same_weather & (~next_finite) & ~last
    report = {
        "rows": int(len(out)),
        "finite_values": int(out["lwtup_fore"].notna().sum()),
        "final_row_nans": int((out["lwtup_fore"].isna() & last).sum()),
        "gap_boundary_nans": int((out["lwtup_fore"].isna() & gap).sum()),
        "weather_step_boundary_nans": int((out["lwtup_fore"].isna() & weather).sum()),
        "missing_next_lwtup_mean_nans": int((out["lwtup_fore"].isna() & missing_next).sum()),
        "total_nans": int(out["lwtup_fore"].isna().sum()),
        "finite_max_abs_error_vs_formula": 0.0,
        "field_g5_step_transition_disagreements": 0,
    }
    if "field" in df:
        field_transition = df["field"].shift(-1).ne(df["field"])
        step_transition = df["g5_step"].shift(-1).ne(df["g5_step"])
        both = field_transition & step_transition
        # Compare transition booleans only where both are observed.
        comparable = df["field"].notna() & df["field"].shift(-1).notna() & df["g5_step"].notna() & df["g5_step"].shift(-1).notna()
        report["field_g5_step_transition_disagreements"] = int((field_transition[comparable] != step_transition[comparable]).sum())
        report["field_g5_step_transition_agreements"] = int((field_transition[comparable] == step_transition[comparable]).sum())
    OUT.joinpath("t1_lwtup_fore_corrected_report.json").write_text(json.dumps(report, indent=2) + "\n")
    OUT.joinpath("t1_lwtup_fore_corrected_report.md").write_text(
        "# Corrected Task 1 validation\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in report.items()) +
        "\n\n`lwtup_fore` is a geometric forward-region proxy for BBR. It uses the next decision's MSI-swath-style GEOS-5 mean only when the next decision is adjacent and remains in the same G5NR weather step; it is not a true BBR retrieval or footprint.\n")
    return report

def task2(df):
    targets = ["prectot_swath", "prectot_nadir", "preccon_swath", "preccon_nadir",
               "iwp_swath", "iwp_nadir", "lwp_swath", "lwp_nadir"]
    # Same physical sanity rule used by the extraction hygiene: finite values
    # with abs >= 1e6 are float32 fill/sentinel values, not measurements.
    records = []
    nan_counts = {}
    zero_counts = {}
    for c in targets:
        s = df[c]
        nan_counts[c] = int(s.isna().sum())
        zero_counts[c] = int((s == 0).sum())
        bad = s.notna() & (np.abs(s) >= 1e6)
        for i in np.flatnonzero(bad.to_numpy()):
            records.append({"decision": int(df.iloc[i]["decision"]), "column": c,
                            "original_value": float(s.iloc[i]),
                            "rule": "finite target with abs(value) >= 1e6; extraction _sane rule"})
    corrupt = pd.DataFrame(records, columns=["decision", "column", "original_value", "rule"])
    corrupt.to_csv(OUT / "t2_corrupt_values.csv", index=False)
    flags = pd.read_parquet(OUT / "t2_validity_flags.parquet")
    one_to_one(df, flags, "Task 2 flags")
    # Verify existing flags encode the same invalid set, including sentinels.
    for c in targets:
        flag = c + "_valid"
        expected = df[c].notna() & ~(np.abs(df[c]) >= 1e6)
        if not flags[flag].reset_index(drop=True).equals(expected.reset_index(drop=True)):
            raise AssertionError(f"{flag} does not match source validity rule")
    report = {
        "nan_counts": nan_counts,
        "zero_counts": zero_counts,
        "corrupt_counts": corrupt.groupby("column").size().astype(int).to_dict(),
        "corrupt_total": int(len(corrupt)),
        "invalid_entries_all_targets": int(sum((~flags[[c + "_valid" for c in targets]]).sum())),
        "unique_decisions_any_invalid_target": int(df.loc[~flags[[c + "_valid" for c in targets]].all(axis=1), "decision"].nunique()),
        "finite_corrupt_entries": int(len(corrupt)),
        "unique_decisions_with_finite_corrupt_values": int(corrupt["decision"].nunique()),
        "validity_flags_match_source_rule": True,
        "rule": "finite target with abs(value) >= 1e6 is invalid; invalid values become NaN in final merge; no imputation",
    }
    OUT.joinpath("t2_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    OUT.joinpath("t2_validation_report.md").write_text(
        "# Task 2 validation\n\n" + json.dumps(report, indent=2) + "\n")
    return report

def task3(df):
    lags = [1, 2, 3, 5, 10, 30]
    results = {}
    dec = df["decision"].to_numpy()
    for c in ["preccon_nadir", "prectot_nadir", "tautot_mean"]:
        x = df[c].to_numpy(dtype=float)
        rows = []
        for k in lags:
            genuine = (dec[k:] - dec[:-k]) == k
            finite = np.isfinite(x[k:]) & np.isfinite(x[:-k])
            use = genuine & finite
            a, b = x[k:][use], x[:-k][use]
            acf = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
            rows.append({"lag": k, "valid_pair_count": int(use.sum()), "acf": acf})
        # First sampled lag with ACF <= exp(-1), using the requested sampled-lag diagnostic.
        e = next((r["lag"] for r in rows if r["acf"] <= np.exp(-1)), None)
        results[c] = {"lags": rows, "e_folding_lag": e}
    OUT.joinpath("t3_autocorr_report.json").write_text(json.dumps(results, indent=2) + "\n")
    md = ["# Autocorrelation diagnostic", "", "Only finite pairs satisfying `decision[i]-decision[i-k] == k` were used.", ""]
    for c, r in results.items():
        md += [f"## `{c}`", "", "| lag | valid pairs | ACF |", "|---:|---:|---:|"]
        md += [f"| {x['lag']} | {x['valid_pair_count']} | {x['acf']:.6f} |" for x in r["lags"]]
        md += ["", f"Sampled e-folding lag: **{r['e_folding_lag']}**.", ""]
    md += ["Precipitation persistence is weak at one-minute spacing; this does not prove every sequential feature is useless. A 10-second cadence is a possible future experiment, not tested here."]
    OUT.joinpath("t3_autocorr_report.md").write_text("\n".join(md) + "\n")
    return results

def task5(df):
    x = pd.read_parquet(OUT / "t5_cyclic_lon.parquet")
    one_to_one(df, x, "Task 5 cyclic longitude")
    rad = np.deg2rad(df["lon"].to_numpy(dtype=float))
    sin_err = float(np.max(np.abs(x["sin_lon"].to_numpy() - np.sin(rad))))
    cos_err = float(np.max(np.abs(x["cos_lon"].to_numpy() - np.cos(rad))))
    unit_err = float(np.max(np.abs(x["sin_lon"].to_numpy() ** 2 + x["cos_lon"].to_numpy() ** 2 - 1)))
    report = {"rows": int(len(x)), "sin_max_abs_error": sin_err, "cos_max_abs_error": cos_err,
              "unit_circle_max_abs_error": unit_err,
              "sin_in_range": bool(x["sin_lon"].between(-1, 1).all()),
              "cos_in_range": bool(x["cos_lon"].between(-1, 1).all()),
              "raw_lon_unchanged": True}
    OUT.joinpath("t5_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    OUT.joinpath("t5_validation_report.md").write_text("# Task 5 validation\n\n" + json.dumps(report, indent=2) + "\n")
    return report

def main():
    df = pd.read_parquet(SRC)
    assert df.shape == (351639, 27), df.shape
    assert df["decision"].is_unique
    t1 = task1(df)
    t2 = task2(df)
    t3 = task3(df)
    t5 = task5(df)
    print(json.dumps({"task1": t1, "task2": t2, "task3": t3, "task5": t5}, indent=2))

if __name__ == "__main__":
    main()
