"""Run the baseline-paper experiments for one mission.

    python -m baselines.run_baselines --mission earthcare

Reads the mission's dataset (missions/<mission>/data/) and writes one CSV per result table, two
figures, and run_meta.json (dataset hash, package versions, seed) to missions/<mission>/results/;
--data and --out override those paths. Every table and figure in the paper must come from these files.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from . import experiments as E
import lightgbm

from .core import SEED, load
from .missions import get_mission

ALL = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plot_curves(df, group, title, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, g in df.groupby(group, sort=False):
        ref = name in ("P1 random", "P8 oracle")
        ax.plot(100 * g["d"], 100 * g["capture"], ("--" if ref else "-"),
                color=("0.55" if ref else None), lw=1.4, label=name)
    ax.set(xlabel="duty cycle d (%)", ylabel="capture@d (%)", xlim=(0, 100), ylim=(0, 101),
           title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", default="earthcare")
    ap.add_argument("--data", help="default: the mission's data_path")
    ap.add_argument("--out", help="default: the mission's results_dir")
    ap.add_argument("--experiments", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--mlp", action="store_true", help="include P7 (small MLP) in E3/E9")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    mission = get_mission(a.mission)
    a.data = a.data or mission.data_path
    out = Path(a.out or mission.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.data, mission)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    log(f"{mission.name}: {len(df):,} rows from {a.data}")

    meta = {"mission": mission.name, "data": str(a.data), "data_sha256": sha256(a.data),
            "rows": len(df), "seed": SEED, "experiments": a.experiments,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "label": f"{mission.label_col} >= {mission.threshold:.6g} kg m-2 s-1",
            "split": {"train": mission.train_months, "val": mission.val_months,
                      "test": mission.test_months}}
    save = lambda t, name: (t.to_csv(out / f"{name}.csv", index=False), log(f"wrote {name}.csv"))

    need_fitted = {"e9", "e10", "e11"} & set(a.experiments)
    fitted = {}
    if "e1" in a.experiments:
        save(E.e1_split(df, mission), "e1_split")
    if "e2" in a.experiments:
        save(E.e2_lomo(df, mission), "e2_leave_one_month_out")
    if "e3" in a.experiments or need_fitted:
        table, curves, (_, _, _, scores) = E.e3_capture(df, mission, include_mlp=a.mlp)
        fitted = {k: v[0] for k, v in scores.items()}
        if "e3" in a.experiments:
            save(table, "e3_capture_table")
            save(curves, "e3_capture_curves")
            plot_curves(curves, "policy", f"{mission.name}: capture vs duty cycle (test block)",
                        out / "fig3_capture_curves.png")
    if "e4" in a.experiments:
        grid, extra = E.e4_label_geometry(df, mission)
        save(grid, "e4_label_geometry")
        meta["e4"] = extra
    if "e5" in a.experiments:
        tiers, night = E.e5_tiers(df, mission)
        save(tiers, "e5_tiers")
        save(night, "e5_night_curves")
        plot_curves(night.rename(columns={"tier": "policy"}), "policy",
                    f"{mission.name}: capture vs duty cycle by tier, night only",
                    out / "fig5_tiers_night.png")
    if "e6" in a.experiments:
        save(E.e6_strata(df, mission), "e6_strata")
    if "e7" in a.experiments:
        save(E.e7_labels(df, mission), "e7_label_sensitivity")
    if "e8" in a.experiments:
        save(E.e8_missing(df, mission), "e8_missing_data")
    if "e9" in a.experiments:
        save(E.e9_cost(df, mission, fitted), "e9_policy_cost")
    if "e10" in a.experiments:
        keep = {k: v for k, v in fitted.items() if k.split()[0] in ("P2", "P4", "P5", "P6")}
        save(E.e10_bootstrap(df, mission, keep, n_boot=a.n_boot), "e10_bootstrap")
    if "e11" in a.experiments:
        best, perm, single = E.e11_importance(df, mission, fitted)
        save(perm, "e11_permutation_importance")
        save(single, "e11_single_feature_auc")
        meta["e11_best_policy"] = best
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log("done")


if __name__ == "__main__":
    main()
