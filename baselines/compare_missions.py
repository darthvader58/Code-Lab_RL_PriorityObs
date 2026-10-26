"""E12: compare the missions and test transfer between them (docs/baseline_paper/07_experiments.md).

    python -m baselines.compare_missions

Reads the EarthCARE and TRMM datasets and writes the E12 tables (CSV) and run_meta.json to
missions/comparison/results/ and Fig. 7 to docs/baseline_paper/figures/comparison/. Every comparison
is restricted to the latitudes both missions sample (|lat| below TRMM's highest latitude), for
training as well as testing, so the orbit is compared and not the climate zone.

- e12_comparison: each mission's own policy side by side (base rate, swath-label inflation, AUC,
  capture@d, Tier B retention).
- e12_transfer: train on one mission, test on the other with the features both have. Two label
  settings: "nadir" uses prectot_nadir >= 1 mm/hr on both sides, so only the sampling differs;
  "own radar" trains on the source mission's label and scores the target's (a deployed gate).
  Thresholds come from the target's unlabeled validation scores (a duty budget needs no labels).
- e12_local_time: TRMM's own policy by local solar hour, and inside vs outside EarthCARE's two
  sampling windows (±1 h of 01:56 and 13:56), each with its own budget threshold; EarthCARE's own
  policy for comparison.
- e12_diurnal: TRMM's rain occurrence and amount by local hour, and the share of daily rain that
  falls outside EarthCARE's windows (TRMM samples every hour almost equally).
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
import lightgbm

from .core import BUDGETS, SEED, LearnedPolicy, blocked_split, gate_metrics, load, threshold_for, valid
from .experiments import label_fn
from .missions import EARTHCARE, GEOMETRY, MM_PER_HR, ROOT, TRMM
from .paper_figures import GRID, INK, INK2, REF, SLOTS, pct

OUT = ROOT / "missions" / "comparison" / "results"
FIG = ROOT / "docs" / "baseline_paper" / "figures" / "comparison"
EC_NODES = (1 + 56 / 60, 13 + 56 / 60)     # EarthCARE's local times at the nodes (Limitation 8)
EC_HALF_WIDTH = 1.0                         # holds every EarthCARE minute in the shared band
D_MAIN = 0.20
CURVE = tuple(np.round(np.linspace(0.01, 1, 100), 2))
# the cloud proxies both imagers give (EarthCARE's Tier A without its forward view) + geometry
SHARED = {"shared cloud + geometry": TRMM.tiers["A"], "geometry only (C)": GEOMETRY}
MISSIONS = {"EarthCARE": EARTHCARE, "TRMM": TRMM}
COLOR = {"EarthCARE": SLOTS[0], "TRMM": SLOTS[1]}


def in_ec_window(hour):
    h = np.asarray(hour, float)
    return np.logical_or.reduce([np.abs((h - n + 12) % 24 - 12) < EC_HALF_WIDTH for n in EC_NODES])


def blocks(df, mission, cols):
    """Train/val/test (blocked split of the mission) over rows whose `cols` are all valid."""
    ok = np.logical_and.reduce([valid(df, c) for c in cols])
    part = blocked_split(df, mission)
    return {p: df[ok & (part == p)] for p in ("train", "val", "test")}


def score_rows(s_val, s_test, y, mass, budgets=BUDGETS):
    return [{"d": d, **gate_metrics(s_test >= threshold_for(s_val, d), y, mass, s_test)}
            for d in budgets]


def oracle_capture(y, d):
    """Best possible capture@d: every firing on a positive."""
    return min(1.0, d / max(np.mean(y), 1e-12))


# ---- 1. side-by-side comparison --------------------------------------------------------------

def e12_comparison(data):
    rows, curves = [], []
    for name, (m, df) in data.items():
        b = blocks(df, m, [m.label_col, m.swath_col])
        lab = label_fn(m.label_col, m.threshold)
        y = lab(b["test"])
        swath = label_fn(m.swath_col, m.threshold)(b["test"]).mean()
        cap = {}
        for pname, kind, tier in (("P5 logistic regression", "lr", "A"),
                                  ("P6 gradient boosting", "lgbm", "A"),
                                  ("P6 gradient boosting, Tier B", "lgbm", "B")):
            pol = LearnedPolicy(kind, m.tiers[tier]).fit(b["train"], lab(b["train"]))
            s_val, s_test = pol.score(b["val"]), pol.score(b["test"])
            res = {r["d"]: r for r in score_rows(s_val, s_test, y, b["test"][m.mass_col], CURVE)}
            cap[pname] = res[D_MAIN]["capture"]
            if tier == "A":
                curves += [{"mission": name, "policy": pname, "d": d, "capture": r["capture"],
                            "oracle": oracle_capture(y, d)} for d, r in res.items()]
                r20, r13 = res[D_MAIN], res[0.13]
                rows.append({"mission": name, "policy": pname, "n_test": len(y),
                             "base_rate": y.mean(), "swath_rate": swath,
                             "swath_inflation": swath / y.mean(), "auc": r20["auc"],
                             "ap": r20["ap"], "capture@13%": r13["capture"],
                             "capture@20%": r20["capture"],
                             "oracle_capture@20%": oracle_capture(y, D_MAIN),
                             "mass_capture@20%": r20["mass_capture"]})
        for r in rows:
            if r["mission"] == name:
                r["tier_b_retention@20%"] = (cap["P6 gradient boosting, Tier B"]
                                             / cap["P6 gradient boosting"])
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---- 2. transfer -----------------------------------------------------------------------------

def e12_transfer(data):
    nadir = "prectot_nadir"
    rows = []
    for setting in ("nadir", "own radar"):
        col = {n: (nadir if setting == "nadir" else m.label_col) for n, (m, _) in data.items()}
        b = {n: blocks(df, m, [col[n]]) for n, (m, df) in data.items()}
        for feats_name, feats in SHARED.items():
            for kind in ("lr", "lgbm"):
                for src in data:
                    lab_src = label_fn(col[src], MM_PER_HR)
                    pol = LearnedPolicy(kind, feats).fit(b[src]["train"], lab_src(b[src]["train"]))
                    for dst, (m_dst, _) in data.items():
                        te, va = b[dst]["test"], b[dst]["val"]
                        y = label_fn(col[dst], MM_PER_HR)(te)
                        res = {r["d"]: r for r in score_rows(pol.score(va), pol.score(te), y,
                                                             te[col[dst]], (0.13, D_MAIN))}
                        rows.append({"label": setting, "features": feats_name,
                                     "model": "P5 logistic regression" if kind == "lr"
                                     else "P6 gradient boosting",
                                     "trained_on": src, "tested_on": dst,
                                     "train_label": col[src], "test_label": col[dst],
                                     "auc": res[D_MAIN]["auc"],
                                     "capture@13%": res[0.13]["capture"],
                                     "capture@20%": res[D_MAIN]["capture"]})
    t = pd.DataFrame(rows)
    key = ["label", "features", "model", "tested_on"]
    own = t[t.trained_on == t.tested_on].set_index(key)
    ref = own.loc[pd.MultiIndex.from_frame(t[key])]
    t["own_capture@20%"] = ref["capture@20%"].to_numpy()
    t["own_auc"] = ref["auc"].to_numpy()
    t["capture_loss_points@20%"] = 100 * (t["own_capture@20%"] - t["capture@20%"])
    t["relative_capture@20%"] = t["capture@20%"] / t["own_capture@20%"]
    return t


# ---- 3. local time ---------------------------------------------------------------------------

def _by_group(pol, b, groups, lab, mass_col, label_name, model):
    """AUC and capture@20% within each local-time group, each with its own val threshold."""
    rows = []
    s_val, s_test = pol.score(b["val"]), pol.score(b["test"])
    for gname, g_val, g_test in groups:
        y = lab(b["test"])[g_test]
        if y.sum() == 0 or y.all():
            continue
        r = score_rows(s_val[g_val], s_test[g_test], y, b["test"][mass_col].to_numpy()[g_test],
                       (D_MAIN,))[0]
        rows.append({"label": label_name, "model": model, "group": gname, "n_test": len(y),
                     "base_rate": y.mean(), "auc": r["auc"], "capture@20%": r["capture"],
                     "oracle_capture@20%": oracle_capture(y, D_MAIN),
                     "lift@20%": r["capture"] / D_MAIN})
    return rows


def window_gap(s_val, s_test, wv, wt, y, day, n_boot=1000, seed=SEED):
    """capture@20% inside minus outside EarthCARE's windows, each with its own val threshold, and a
    95% interval from resampling test days (decisions on one day share weather)."""
    days, di = np.unique(day, return_inverse=True)
    parts = []
    for g_val, g_test in ((wv, wt), (~wv, ~wt)):
        fire = s_test[g_test] >= threshold_for(s_val[g_val], D_MAIN)
        yy = y[g_test]
        parts.append((np.bincount(di[g_test], weights=yy, minlength=len(days)),
                      np.bincount(di[g_test], weights=yy & fire, minlength=len(days))))
    w = np.random.default_rng(seed).multinomial(len(days), np.full(len(days), 1 / len(days)),
                                                size=n_boot)
    cap = [hit.sum() / pos.sum() for pos, hit in parts]
    boot = [(w @ hit) / (w @ pos) for pos, hit in parts]
    lo, hi = np.percentile(boot[0] - boot[1], [2.5, 97.5])
    return {"capture_in_windows": cap[0], "capture_other_hours": cap[1],
            "gap_points": 100 * (cap[0] - cap[1]), "gap_ci_low": 100 * lo, "gap_ci_high": 100 * hi,
            "n_test_days": len(days)}


def e12_local_time(data):
    rows, gaps = [], []
    m, df = data["TRMM"]
    for label_name, col, mass in (("TRMM radar", m.label_col, m.mass_col),
                                  ("nadir", "prectot_nadir", "prectot_nadir")):
        b = blocks(df, m, [col])
        lab = label_fn(col, MM_PER_HR)
        hv, ht = b["val"]["solar_hour"].to_numpy(), b["test"]["solar_hour"].to_numpy()
        wv, wt = in_ec_window(hv), in_ec_window(ht)
        groups = [("EarthCARE windows", wv, wt), ("other hours", ~wv, ~wt),
                  ("all hours", np.ones_like(wv), np.ones_like(wt))]
        groups += [(f"hour {h:02d}", np.floor(hv) == h, np.floor(ht) == h) for h in range(24)]
        for model, kind in (("P5 logistic regression", "lr"), ("P6 gradient boosting", "lgbm")):
            pol = LearnedPolicy(kind, m.tiers["A"]).fit(b["train"], lab(b["train"]))
            rows += [{"mission": "TRMM", **r}
                     for r in _by_group(pol, b, groups, lab, mass, label_name, model)]
            gaps.append({"label": label_name, "model": model,
                         **window_gap(pol.score(b["val"]), pol.score(b["test"]), wv, wt,
                                      lab(b["test"]), b["test"]["g5_day"].to_numpy())})
    m, df = data["EarthCARE"]
    b = blocks(df, m, [m.label_col])
    lab = label_fn(m.label_col, m.threshold)
    for model, kind in (("P5 logistic regression", "lr"), ("P6 gradient boosting", "lgbm")):
        pol = LearnedPolicy(kind, m.tiers["A"]).fit(b["train"], lab(b["train"]))
        ones_v, ones_t = np.ones(len(b["val"]), bool), np.ones(len(b["test"]), bool)
        rows += [{"mission": "EarthCARE", **r} for r in _by_group(
            pol, b, [("EarthCARE windows", ones_v, ones_t)], lab, m.mass_col, "nadir", model)]
    return pd.DataFrame(rows), pd.DataFrame(gaps)


# ---- 4. diurnal cycle ------------------------------------------------------------------------

def e12_diurnal(data):
    """TRMM's rain by local hour over the whole window; hours are sampled almost equally, so hourly
    means can be summed into a daily total."""
    m, df = data["TRMM"]
    d = df[valid(df, m.label_col) & valid(df, "prectot_nadir") & valid(df, m.mass_col)].copy()
    d["hour"] = np.floor(d["solar_hour"]).astype(int)
    d["surface"] = np.where(d["land_frac_nadir"] >= 0.5, "land", "ocean")
    hourly, summary = [], []
    for surf, g in (("all", d), *d.groupby("surface")):
        h = g.groupby("hour").agg(n=("hour", "size"),
                                  radar_rain=(m.label_col, lambda x: (x >= MM_PER_HR).mean()),
                                  nadir_rain=("prectot_nadir", lambda x: (x >= MM_PER_HR).mean()),
                                  rain_amount=(m.mass_col, "mean")).reset_index()
        h["ec_window"] = in_ec_window(h["hour"] + 0.5)
        hourly += [{"surface": surf, **r} for r in h.to_dict("records")]
        w = h["ec_window"]
        summary.append({"surface": surf, "n": int(h["n"].sum()),
                        "window_share_of_hours": w.mean(),
                        "rain_amount_outside_windows": h.loc[~w, "rain_amount"].sum()
                        / h["rain_amount"].sum(),
                        "radar_rain_in_windows": h.loc[w, "radar_rain"].mean(),
                        "radar_rain_other_hours": h.loc[~w, "radar_rain"].mean(),
                        "nadir_rain_in_windows": h.loc[w, "nadir_rain"].mean(),
                        "nadir_rain_other_hours": h.loc[~w, "nadir_rain"].mean(),
                        "peak_hour_rain_amount": int(h.loc[h["rain_amount"].idxmax(), "hour"])})
    return pd.DataFrame(hourly), pd.DataFrame(summary)


# ---- Fig. 7 ----------------------------------------------------------------------------------

def shade_windows(ax):
    for n in EC_NODES:
        ax.axvspan(n - EC_HALF_WIDTH, n + EC_HALF_WIDTH, color=GRID, alpha=0.8, lw=0, zorder=0)


def fig7(curves, local, hourly, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for name, g in curves[curves.policy == "P6 gradient boosting"].groupby("mission", sort=False):
        ax.plot(g["d"], g["capture"], color=COLOR[name], lw=2, label=f"{name} gradient boosting")
        ax.plot(g["d"], g["oracle"], "--", color=COLOR[name], lw=1.1, label=f"{name} oracle")
    ax.plot([0, 1], [0, 1], ":", color=REF, lw=1.2, label="random")
    ax.axvline(D_MAIN, color=INK2, lw=0.8, ls=":")
    ax.set(xlabel="duty cycle d", ylabel="capture@d (own radar label)", xlim=(0, 1), ylim=(0, 1.01),
           title="a  Capture by duty cycle, |lat| < 35°")
    pct(ax), pct(ax, "x")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    t = local[(local.mission == "TRMM") & (local.model == "P6 gradient boosting")
              & local.group.str.startswith("hour")].copy()
    t["h"] = t.group.str[5:].astype(int) + 0.5
    shade_windows(ax)
    for (lname, c) in (("TRMM radar", COLOR["TRMM"]), ("nadir", SLOTS[2])):
        g = t[t.label == lname]
        ax.plot(g["h"], g["auc"], "-o", ms=3.5, color=c, lw=1.6,
                label=f"TRMM, {'radar-swath' if lname == 'TRMM radar' else 'nadir'} label")
    ec = local[(local.mission == "EarthCARE") & (local.model == "P6 gradient boosting")]
    if len(ec):
        ax.hlines([ec["auc"].iloc[0]] * len(EC_NODES), [n - EC_HALF_WIDTH for n in EC_NODES],
                  [n + EC_HALF_WIDTH for n in EC_NODES], color=COLOR["EarthCARE"], lw=3,
                  label="EarthCARE, nadir label")
    lo = min(t["auc"].min(), ec["auc"].min() if len(ec) else 1)
    ax.set(xlabel="local solar hour", ylabel="AUC (gradient boosting)", xlim=(0, 24),
           ylim=(lo - 0.025, None), xticks=range(0, 25, 3),
           title="b  Skill by local hour (EarthCARE windows shaded)")
    ax.legend(fontsize=7, loc="lower center", ncol=3, handlelength=1.5, columnspacing=1)

    ax = axes[2]
    h = hourly[hourly.surface.isin(["land", "ocean"])]
    shade_windows(ax)
    for surf, c in (("land", SLOTS[3]), ("ocean", SLOTS[0])):
        g = h[h.surface == surf]
        ax.plot(g["hour"] + 0.5, g["nadir_rain"], "-o", ms=3.5, color=c, lw=1.6, label=surf)
    ax.set(xlabel="local solar hour", ylabel="rain ≥ 1 mm/hr at nadir (share of minutes)",
           xlim=(0, 24), xticks=range(0, 25, 3), ylim=(0, None),
           title="c  TRMM rain occurrence by local hour")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    pct(ax)
    ax.legend(fontsize=7.5)
    for a in axes:
        a.title.set_color(INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---- runner ----------------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fig", default=str(FIG))
    a = ap.parse_args(argv)
    out, fig = Path(a.out), Path(a.fig)
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    save = lambda t, name: (t.to_csv(out / f"{name}.csv", index=False), log(f"wrote {name}.csv"))

    full = {n: load(m.data_path, m) for n, m in MISSIONS.items()}
    lat_max = float(full["TRMM"]["lat"].abs().max())
    data = {n: (m, full[n][full[n]["lat"].abs() <= lat_max].reset_index(drop=True))
            for n, m in MISSIONS.items()}
    for n, (m, d) in data.items():
        log(f"{n}: {len(d):,} of {len(full[n]):,} decisions with |lat| <= {lat_max:.2f}")

    table, curves = e12_comparison(data)
    save(table, "e12_comparison")
    save(curves, "e12_capture_curves")
    save(e12_transfer(data), "e12_transfer")
    local, gaps = e12_local_time(data)
    save(local, "e12_local_time")
    save(gaps, "e12_local_time_gap")
    hourly, summary = e12_diurnal(data)
    save(hourly, "e12_diurnal_hourly")
    save(summary, "e12_diurnal_summary")
    fig7(curves, local, hourly, fig / "fig7_mission_comparison.png")
    log("wrote fig7_mission_comparison.png")

    meta = {"experiment": "e12", "lat_max": lat_max, "seed": SEED,
            "ec_windows": {"nodes_local_hour": EC_NODES, "half_width_h": EC_HALF_WIDTH},
            "data": {n: {"path": str(m.data_path.relative_to(ROOT)),
                         "sha256": sha256(m.data_path), "rows_in_band": len(data[n][1])}
                     for n, m in MISSIONS.items()},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log("done")


if __name__ == "__main__":
    main()
