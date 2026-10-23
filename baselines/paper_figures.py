"""Draw the paper figures for one mission from its baseline results.

    python -m baselines.paper_figures --mission earthcare

Reads the result CSVs in missions/<mission>/results/ (written by run_baselines) and, for the
label-geometry map, the mission's parquet. Writes PNGs to docs/baseline_paper/figures/<mission>/.
No numbers are typed in by hand: every bar, curve and interval comes from those files.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .missions import ROOT, get_mission

# categorical slots in fixed order (dataviz reference palette, light mode); grey for references
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLICY_COLOR = {"P2 geometry LR": SLOTS[0], "P3 latitude band": SLOTS[1],
                "P4 single threshold": SLOTS[2], "P5 logistic regression": SLOTS[3],
                "P6 gradient boosting": SLOTS[4], "P7 MLP": SLOTS[6]}
TIER_NAME = {"A": "A (full)", "A_notau": "A − optical depth", "B": "B (cheap)",
             "C": "C (geometry only)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, out, name, log):
    fig.savefig(out / name)
    plt.close(fig)
    log.append(name)


def fig_capture(res, out, log):
    c = pd.read_csv(res / "e3_capture_curves.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, g in c.groupby("policy", sort=False):
        if name in ("P1 random", "P8 oracle"):
            ax.plot(g["d"], g["capture"], "--", color=REF, lw=1.2, label=name)
        else:
            ax.plot(g["d"], g["capture"], color=POLICY_COLOR[name], lw=2, label=name)
    best = c[(c.policy == "P6 gradient boosting") & np.isclose(c.d, 0.20)].iloc[0]
    ax.axvline(0.20, color=INK2, lw=0.8, ls=":")
    ax.plot(0.20, best.capture, "o", ms=8, color=POLICY_COLOR["P6 gradient boosting"],
            mec="white", mew=2, zorder=5)
    ax.annotate(f"{best.capture:.1%} at 20% duty", (0.20, best.capture), xytext=(0.23, 0.92),
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="duty cycle d (fraction of minutes the radar is on)",
           ylabel="capture@d (share of rainy minutes caught)",
           title="Capture vs. duty cycle, time-blocked test block (Nov 2005 – Jan 2006)")
    pct(ax), pct(ax, "x")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "fig3_capture_curves.png", log)


def fig_label_geometry(m, res, out, log):
    meta = json.loads((res / "run_meta.json").read_text())["e4"]["positive_rate"]
    g = pd.read_csv(res / "e4_label_geometry.csv")
    d = pd.read_parquet(m.data_path, columns=["lat", "lon", m.label_col, m.swath_col])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})

    # (a) positive rate, footprint vs swath, test block
    ax = axes[0]
    names = list(m.label_geometries)
    x = np.arange(len(names))
    foot = [meta[m.label_geometries[v][0]] for v in names]
    swath = [meta[m.label_geometries[v][1]] for v in names]
    ax.bar(x - 0.2, foot, 0.38, color=SLOTS[0], label="radar footprint (nadir)")
    ax.bar(x + 0.2, swath, 0.38, color=SLOTS[1], label="imager swath (150 km)")
    for i in x:
        ax.text(i - 0.2, foot[i], f"{foot[i]:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(i + 0.2, swath[i], f"{swath[i]:.1%}", ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.set_xticks(x, ["total ≥ 1 mm/hr", "convective ≥ 1 mm/hr"])
    ax.set(title="(a) Positive rate", ylim=(0, max(swath) * 1.25))
    pct(ax)
    ax.legend(fontsize=7.5, loc="upper right")

    # (b) precision@20% of the nadir-trained policy, scored on each label
    ax = axes[1]
    own = g[g.trained_on == g.scored_on.str.replace("swath", "nadir")]
    own = own[own.trained_on.str.endswith("nadir")]
    for i, v in enumerate(names):
        r = own[own.variable == v].set_index("scored_on")["precision@20%"]
        p_f, p_s = r[m.label_geometries[v][0]], r[m.label_geometries[v][1]]
        ax.bar(i - 0.2, p_f, 0.38, color=SLOTS[0])
        ax.bar(i + 0.2, p_s, 0.38, color=SLOTS[1])
        ax.text(i - 0.2, p_f, f"{p_f:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(i + 0.2, p_s, f"{p_s:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, ["total", "convective"])
    ax.set(title="(b) Precision at 20% duty, same policy", ylim=(0, 0.9))
    pct(ax)

    # (c) nadir positive rate by latitude, whole dataset
    ax = axes[2]
    bins = np.arange(-90, 91, 5)
    lat_bin = pd.cut(d["lat"], bins)
    prof = pd.DataFrame({"nadir": d[m.label_col] >= m.threshold,
                         "swath": d[m.swath_col] >= m.threshold}).groupby(lat_bin, observed=True).mean()
    mid = [b.mid for b in prof.index]
    ax.plot(mid, prof["nadir"], color=SLOTS[0], lw=2, label="radar footprint (nadir)")
    ax.plot(mid, prof["swath"], color=SLOTS[1], lw=2, label="imager swath")
    ax.fill_between(mid, prof["nadir"], prof["swath"], color=SLOTS[1], alpha=0.12, lw=0)
    ax.set(title="(c) Positive rate by latitude, all 352,800 decisions", xlabel="latitude (°)",
           xlim=(-85, 85))
    pct(ax)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig4_label_geometry.png", log)

    # Fig. 2: where the radar sees rain (5° cells), with orbit sampling density
    cell = 5
    d["ilat"] = (np.floor(d["lat"] / cell) * cell).astype(int)
    d["ilon"] = (np.floor(((d["lon"] + 180) % 360 - 180) / cell) * cell).astype(int)
    grid = d.assign(pos=d[m.label_col] >= m.threshold).groupby(["ilat", "ilon"])["pos"].agg(
        ["mean", "size"]).reset_index()
    grid = grid[grid["size"] >= 30]
    lats, lons = np.arange(-90, 90, cell), np.arange(-180, 180, cell)
    z = np.full((len(lats), len(lons)), np.nan)
    z[((grid.ilat + 90) // cell).values, ((grid.ilon + 180) // cell).values] = grid["mean"].values
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
    cmap.set_bad("#f1f0ec")
    im = ax.pcolormesh(np.r_[lons, 180], np.r_[lats, 90], np.ma.masked_invalid(z), cmap=cmap,
                       vmin=0, vmax=0.4, shading="flat")
    ax.set(xlabel="longitude (°)", ylabel="latitude (°)", xlim=(-180, 180), ylim=(-85, 85),
           title="Share of decisions with ≥ 1 mm/hr under the radar (5° cells, ≥ 30 decisions)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="max")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    save(fig, out, "fig2_rain_map.png", log)


def fig_tiers(res, out, log):
    n = pd.read_csv(res / "e5_night_curves.csv")
    t = pd.read_csv(res / "e5_tiers.csv")
    colors = {"A": SLOTS[0], "A_notau": SLOTS[1], "B": SLOTS[2], "C": SLOTS[3]}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for tier, g in n.groupby("tier", sort=False):
        ax.plot(g["d"], g["capture"], color=colors[tier], lw=2, label=TIER_NAME[tier])
    ax.plot([0, 1], [0, 1], "--", color=REF, lw=1.2, label="random")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="duty cycle d", ylabel="capture@d",
           title="(a) Night only: capture vs. duty cycle by feature tier")
    pct(ax), pct(ax, "x")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    x = np.arange(len(t))
    cols = [("day_capture@20%", "day", SLOTS[0]), ("night_capture@20%", "night", SLOTS[6]),
            ("switched_capture@20%", "day/night switched", SLOTS[2])]
    for k, (col, lab, c) in enumerate(cols):
        vals = t[col].values
        ax.bar(x + (k - 1) * 0.27, np.nan_to_num(vals), 0.25, color=c, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * 0.27, (0 if np.isnan(v) else v) + 0.01,
                    "n/a" if np.isnan(v) else f"{v:.0%}", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x, [TIER_NAME[s] for s in t.tier])
    ax.set(ylim=(0, 0.8), title="(b) Capture at 20% duty, LightGBM")
    pct(ax)
    ax.legend(fontsize=7.5, loc="upper right", ncols=3)
    fig.tight_layout()
    save(fig, out, "fig5_tiers_day_night.png", log)


def fig_strata(res, out, log):
    s = pd.read_csv(res / "e6_strata.csv").iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    y = np.arange(len(s))
    for ax, col, title, lim, fmt in [
            (axes[0], "base_rate", "(a) Base rate", (0, 0.25), "{:.1%}"),
            (axes[1], "auc", "(b) ROC-AUC", (0.8, 1.0), "{:.3f}"),
            (axes[2], "capture@20%", "(c) Capture, global 20% threshold", (0, 1), "{:.1%}")]:
        ax.hlines(y, lim[0], s[col], color=GRID, lw=1.5)
        ax.plot(s[col], y, "o", ms=8, color=SLOTS[0], mec="white", mew=2)
        for yi, v in zip(y, s[col]):
            ax.text(v, yi + 0.28, fmt.format(v), ha="center", fontsize=7.5, color=INK)
        ax.set(xlim=lim, title=title)
        ax.grid(axis="y", visible=False)
        if col != "auc":
            pct(ax, "x")
    ax = axes[2]
    ax.plot(s["duty"], y, "|", ms=12, mew=2, color=SLOTS[1], label="duty used in stratum")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    axes[0].set_yticks(y, s["stratum"])
    axes[0].set_ylim(-0.5, len(s) - 0.3)
    axes[0].axhline(2.5, color=INK2, lw=0.6)
    fig.tight_layout()
    save(fig, out, "fig6_strata.png", log)


def fig_supplementary(m, res, out, log):
    # bootstrap intervals
    b = pd.read_csv(res / "e10_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, metric, title in [(axes[0], "auc", "(a) ROC-AUC"),
                              (axes[1], "capture@20%", "(b) Capture at 20% duty")]:
        r = b[b.metric == metric].iloc[::-1]
        y = np.arange(len(r))
        for yi, (_, row) in zip(y, r.iterrows()):
            c = POLICY_COLOR[row.policy]
            ax.plot([row.ci_low, row.ci_high], [yi, yi], color=c, lw=2)
            ax.plot(row["mean"], yi, "o", ms=7, color=c, mec="white", mew=1.5)
            ax.text(row.ci_high, yi + 0.22, f"{row['mean']:.3f}" if metric == "auc"
                    else f"{row['mean']:.1%}", fontsize=7.5, color=INK)
        ax.set_yticks(y, r.policy)
        ax.set_ylim(-0.5, len(r) - 0.3)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        if metric != "auc":
            pct(ax, "x")
    fig.suptitle("95% intervals, whole-day block bootstrap of the test block (n = 1,000)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, out, "figS1_bootstrap.png", log)

    # importance
    p = pd.read_csv(res / "e11_permutation_importance.csv").iloc[::-1]
    a = pd.read_csv(res / "e11_single_feature_auc.csv").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.barh(p.feature, p.auc_drop_mean, xerr=p.auc_drop_std, color=SLOTS[0], height=0.65,
            error_kw=dict(ecolor=INK2, lw=0.8))
    ax.set(title="(a) LightGBM permutation importance", xlabel="drop in test AUC when shuffled")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(a.feature, a.auc - 0.5, left=0.5, color=SLOTS[0], height=0.65)
    ax.axvline(0.5, color=REF, ls="--", lw=1)
    ax.set(title="(b) Single-feature ROC-AUC (* = sign inverted)", xlim=(0.45, 0.9),
           xlabel="AUC")
    ax.set_yticks(range(len(a)), [f + (" *" if inv else "") for f, inv in zip(a.feature, a.inverted)])
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "figS2_feature_importance.png", log)

    # split robustness
    # solid: each month held out, trained on the other seven; dashed: the summer-trained
    # time-blocked test, drawn only over the months it scores
    e2 = pd.read_csv(res / "e2_leave_one_month_out.csv")
    blocked = pd.read_csv(res / "e3_capture_table.csv").set_index("policy")["auc"]
    months = list(dict.fromkeys(e2.held_out))
    test = [i for i, mo in enumerate(months) if mo >= m.test_months[0]]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for p, g in e2.groupby("policy", sort=False):
        c = POLICY_COLOR[p]
        ax.plot(range(len(g)), g.auc, "o-", color=c, lw=2, ms=6, mec="white", mew=1.2,
                label=f"{p.split()[0]} leave-one-month-out")
        ax.hlines(blocked[p], test[0] - 0.3, test[-1] + 0.3, color=c, lw=1.5, ls="--",
                  label=f"{p.split()[0]} trained Jun–Sep only ({blocked[p]:.3f})")
    ax.axvspan(test[0] - 0.5, test[-1] + 0.5, color=REF, alpha=0.08, lw=0)
    ax.set_xticks(range(len(months)), months)
    ax.set(ylim=(0.78, 0.93), ylabel="ROC-AUC", xlabel="held-out month",
           title="Skill by held-out month vs. the summer-trained test block")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, out, "figS3_split_robustness.png", log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", default="earthcare")
    a = ap.parse_args(argv)
    m = get_mission(a.mission)
    res = Path(m.results_dir)
    out = ROOT / "docs" / "baseline_paper" / "figures" / m.name
    out.mkdir(parents=True, exist_ok=True)
    log = []
    fig_capture(res, out, log)
    fig_label_geometry(m, res, out, log)
    fig_tiers(res, out, log)
    fig_strata(res, out, log)
    fig_supplementary(m, res, out, log)
    print(f"wrote {len(log)} figures to {out}:", *log, sep="\n  ")


if __name__ == "__main__":
    main()
