"""Experiments E1-E11 of docs/baseline_paper/07_experiments.md for one mission.

Every function takes the loaded frame and the mission config and returns a DataFrame (or a dict of
them). Nothing here reads or writes files; run_baselines.py does that.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import (BUDGETS, SEED, LatitudeBandPolicy, LearnedPolicy, RandomPolicy,
                   SingleFeaturePolicy, SwitchedPolicy, blocked_split, evaluate, gate_metrics,
                   oracle_rows, random_split, seasonal_folds, threshold_for, valid)

D_MAIN = 0.20          # budget used for single-number comparisons (07_experiments.md E4-E6)


def label_fn(col, thr, strict=False):
    if strict:                    # median split: "> median", so a zero median cannot label every row
        return lambda frame: (frame[col].to_numpy() > thr)
    return lambda frame: (frame[col].to_numpy() >= thr)


def _blocks(df, mission, cols):
    """Blocked split restricted to rows whose target columns are all valid."""
    ok = np.logical_and.reduce([valid(df, c) for c in cols])
    part = blocked_split(df, mission)
    d = df[ok]
    p = part[ok]
    return d[p == "train"], d[p == "val"], d[p == "test"]


def policies(mission, include_mlp=False):
    A, C = mission.tiers["A"], mission.tiers["C"]
    out = {"P1 random": RandomPolicy(), "P2 geometry LR": LearnedPolicy("lr", C),
           "P3 latitude band": LatitudeBandPolicy(), "P4 single threshold": SingleFeaturePolicy(),
           "P5 logistic regression": LearnedPolicy("lr", A),
           "P6 gradient boosting": LearnedPolicy("lgbm", A)}
    if include_mlp:
        out["P7 MLP"] = LearnedPolicy("mlp", A)
    return out


# ---- E1 / E2: splits -------------------------------------------------------------------------

def e1_split(df, mission):
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    d = df[valid(df, col)]
    rows = []
    for name, part in (("random 30%", random_split(d)), ("time-blocked", blocked_split(d, mission))):
        tr, va, te = d[part == "train"], d[part == "val"], d[part == "test"]
        res, _ = evaluate(LearnedPolicy("lr", mission.tiers["A"]), tr, va, te, lab,
                          mission.mass_col, (0.13,))
        rows.append({"split": name, "n_train": len(tr), "n_test": len(te), **res[0]})
    return pd.DataFrame(rows)


def e2_lomo(df, mission, include_mlp=False):
    """Leave one G5NR month out for each learned policy (P5-P7); threshold from the training
    months (no separate val block)."""
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    d = df[valid(df, col)]
    learned = {k: p for k, p in policies(mission, include_mlp).items()
               if k.split()[0] in ("P5", "P6", "P7")}
    rows = []
    for policy, proto in learned.items():
        for fold in seasonal_folds(df):
            m = fold.held_out_month
            tr, te = fold.frames(df, [col])
            pol = LearnedPolicy(proto.kind, proto.features).fit(tr, lab(tr))
            s_tr, s_te = pol.score(tr), pol.score(te)
            base = lab(tr).mean()
            row = {"policy": policy, "held_out": m, "n_test": len(te)}
            for name, dd in (("13%", 0.13), ("base_rate", base)):
                g = gate_metrics(s_te >= threshold_for(s_tr, dd), lab(te), te[col], s_te)
                row.update({"auc": g["auc"], "ap": g["ap"], f"capture@{name}": g["capture"]})
            rows.append(row)
    return pd.DataFrame(rows)


# ---- E3: capture curves ----------------------------------------------------------------------

def e3_capture(df, mission, include_mlp=False, curve=np.round(np.linspace(0.01, 1, 100), 2)):
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    tr, va, te = _blocks(df, mission, [col])
    table, curves, scores = [], [], {}
    for name, pol in policies(mission, include_mlp).items():
        rows, s = evaluate(pol, tr, va, te, lab, mission.mass_col, tuple(curve))
        scores[name] = (pol, s)
        curves += [{"policy": name, **r} for r in rows]
    for name, rows in (("P8 oracle", oracle_rows(te, lab, col, mission.mass_col, tuple(curve))),):
        curves += [{"policy": name, **r} for r in rows]
    curves = pd.DataFrame(curves)
    for name, g in curves.groupby("policy", sort=False):
        row = {"policy": name, "auc": g["auc"].iloc[0], "ap": g["ap"].iloc[0]}
        for d in BUDGETS:
            row[f"capture@{int(d * 100)}%"] = g.loc[np.isclose(g["d"], d), "capture"].iloc[0]
        row["mass_capture@20%"] = g.loc[np.isclose(g["d"], 0.20), "mass_capture"].iloc[0]
        table.append(row)
    return pd.DataFrame(table), curves, (tr, va, te, scores)


# ---- E4: label geometry ----------------------------------------------------------------------

def e4_label_geometry(df, mission, kind="lgbm"):
    """Train/score grid over the mission's label geometries (radar footprint, passive swath, and
    for TRMM the nadir line) for total and convective rain."""
    pairs = mission.label_geometries
    tr, va, te = _blocks(df, mission, [c for p in pairs.values() for c in p])
    rows, rates, share = [], {}, {}
    for variable, cols in pairs.items():
        for train_on in cols:
            pol = LearnedPolicy(kind, mission.tiers["A"])
            pol.fit(tr, label_fn(train_on, mission.threshold)(tr))
            th = threshold_for(pol.score(va), D_MAIN)
            s = pol.score(te)
            for score_on in cols:
                g = gate_metrics(s >= th, label_fn(score_on, mission.threshold)(te), te[score_on], s)
                rows.append({"variable": variable, "trained_on": train_on, "scored_on": score_on,
                             "auc": g["auc"], "capture@20%": g["capture"],
                             "precision@20%": g["precision"], "base_rate": g["base_rate"]})
        footprint_col, swath_col = cols[:2]
        rates.update({c: float((te[c] >= mission.threshold).mean()) for c in cols})
        swath_pos = te[swath_col] >= mission.threshold
        share[variable] = float((swath_pos & (te[footprint_col] < mission.threshold)).sum()
                                / swath_pos.sum())
    return pd.DataFrame(rows), {"positive_rate": rates, "swath_pos_label_neg_share": share}


# ---- E5: feature tiers x day/night -----------------------------------------------------------

def e5_tiers(df, mission, kind="lgbm", curve=np.round(np.linspace(0.01, 1, 100), 2)):
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    tr, va, te = _blocks(df, mission, [col])
    rows, night_curves = [], []
    for tier, feats in mission.tiers.items():
        row = {"tier": tier}
        for regime, sel in (("day", True), ("night", False)):
            if regime == "night" and any(f.startswith("tautot") for f in feats):
                row[f"{regime}_capture@20%"] = np.nan          # tautot is daylight-only
                continue
            m = lambda f: f[f["is_day"] == sel]
            res, _ = evaluate(LearnedPolicy(kind, feats), m(tr), m(va), m(te), lab,
                              mission.mass_col, tuple(curve) if regime == "night" else (D_MAIN,))
            row[f"{regime}_capture@20%"] = [r for r in res if np.isclose(r["d"], D_MAIN)][0]["capture"]
            if regime == "night":
                night_curves += [{"tier": tier, **r} for r in res]
        night_feats = [f for f in feats if not f.startswith("tautot")]
        sw = SwitchedPolicy(LearnedPolicy(kind, feats), LearnedPolicy(kind, night_feats))
        res, _ = evaluate(sw, tr, va, te, lab, mission.mass_col, (D_MAIN,))
        row["switched_capture@20%"] = res[0]["capture"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["retention_vs_A"] = out["switched_capture@20%"] / out.loc[out.tier == "A",
                                                                  "switched_capture@20%"].iloc[0]
    return out, pd.DataFrame(night_curves)


# ---- E6: strata ------------------------------------------------------------------------------

def e6_strata(df, mission, kind="lgbm"):
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    tr, va, te = _blocks(df, mission, [col])
    pol = LearnedPolicy(kind, mission.tiers["A"]).fit(tr, lab(tr))
    th = threshold_for(pol.score(va), D_MAIN)                  # global threshold, not re-tuned
    s = pol.score(te)
    alat = te["lat"].abs()
    strata = {name: (alat >= lo) & (alat < hi) for name, lo, hi in mission.lat_strata}
    strata.update({
        "ocean": te["ocean_frac_nadir"] > 0.95,
        "land": te["land_frac_nadir"] > 0.95,
        "mixed": (te["ocean_frac_nadir"] <= 0.95) & (te["land_frac_nadir"] <= 0.95),
    })
    rows = []
    for name, m in strata.items():
        m = m.to_numpy()
        g = gate_metrics(s[m] >= th, lab(te)[m], te[col].to_numpy()[m], s[m])
        rows.append({"stratum": name, "n": int(m.sum()), "base_rate": g["base_rate"],
                     "auc": g["auc"], "duty": g["duty"], "capture@20%": g["capture"]})
    return pd.DataFrame(rows)


# ---- E7: label sensitivity -------------------------------------------------------------------

def e7_labels(df, mission):
    col, conv, thr = mission.label_col, mission.convective_col, mission.threshold
    variants = [(col, 0.1 * thr, False, f"{col} >= 0.1 mm/hr"),
                (col, thr, False, f"{col} >= 1 mm/hr"), (col, 5 * thr, False, f"{col} >= 5 mm/hr"),
                (conv, thr, False, f"{conv} >= 1 mm/hr")]
    tr0, _, _ = _blocks(df, mission, [col])
    variants.append((col, float(tr0[col].median()), True, f"{col} > train median (median split)"))
    rows = []
    for c, t, strict, name in variants:
        lab = label_fn(c, t, strict)
        tr, va, te = _blocks(df, mission, [c])
        base = lab(tr).mean()
        for kind in ("lr", "lgbm"):
            res, _ = evaluate(LearnedPolicy(kind, mission.tiers["A"]), tr, va, te, lab, c, (base,))
            rows.append({"label": name, "model": kind, "threshold_kg_m2_s": t,
                         "base_rate_test": res[0]["base_rate"], "auc": res[0]["auc"],
                         "ap": res[0]["ap"], "capture@d=base_rate": res[0]["capture"]})
    return pd.DataFrame(rows)


# ---- E8: missing data ------------------------------------------------------------------------

def e8_missing(df, mission):
    col = mission.label_col
    part = blocked_split(df, mission)
    feats = mission.tiers["A"]
    rows = [{"item": "rows in dataset", "count": len(df)},
            {"item": f"rows with invalid {col}", "count": int((~valid(df, col)).sum())},
            {"item": f"rows with invalid {mission.swath_col}",
             "count": int((~valid(df, mission.swath_col)).sum())},
            {"item": "rows with any NaN Tier A feature", "count": int(df[feats].isna().any(axis=1).sum())}]
    rows += [{"item": f"NaN in {f}", "count": int(df[f].isna().sum())} for f in feats
             if df[f].isna().any()]
    ok = valid(df, col)
    for name in ("train", "val", "test", "drop"):
        rows.append({"item": f"{name} rows used (valid label)", "count": int(((part == name) & ok).sum())})
    rows.append({"item": "NaN handling", "count": np.nan,
                 "note": "LR/MLP: median impute + missing indicators; LightGBM: native NaN; P4: NaN scores lowest"})
    return pd.DataFrame(rows)


# ---- E9: policy cost -------------------------------------------------------------------------

def e9_cost(df, mission, fitted):
    """fitted: {name: policy} after E3. Ops are arithmetic estimates, wall time is measured."""
    _, _, te = _blocks(df, mission, [mission.label_col])
    one = te.iloc[[0]]
    rows = []
    for name, pol in ((k, p) for k, p in fitted.items() if isinstance(p, LearnedPolicy)):
        est = pol.model.steps[-1][1]
        n_in = len(pol.features)
        if pol.kind == "lr":
            params = est.coef_.size + est.intercept_.size
            ops = f"~{est.coef_.size} multiply-adds + 1 sigmoid"
        elif pol.kind == "lgbm":
            trees = est.booster_.trees_to_dataframe()
            n_trees = trees["tree_index"].nunique()
            depth = trees.groupby("tree_index")["node_depth"].max().mean() - 1
            params = len(trees)
            ops = f"~{int(round(n_trees * depth))} comparisons ({n_trees} trees, mean depth {depth:.1f})"
        else:
            params = sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_)
            ops = f"~{sum(w.size for w in est.coefs_)} multiply-adds"
        t0 = time.perf_counter()
        for _ in range(200):
            pol.score(one)
        per_call = (time.perf_counter() - t0) / 200
        rows.append({"policy": name, "inputs": n_in, "parameters": int(params), "operations": ops,
                     "serialized_bytes": len(pickle.dumps(pol.model)),
                     "wall_ms_per_decision_python": 1000 * per_call})
    return pd.DataFrame(rows)


# ---- E10: block bootstrap --------------------------------------------------------------------

def e10_bootstrap(df, mission, fitted, n_boot=1000, budgets=(0.13, 0.20), seed=SEED):
    """Resample whole G5NR days of the test block; thresholds stay fixed from validation.

    Every policy is scored on the same draws, so the second table gives paired differences:
    the best policy (highest mean AUC) minus each other policy, per draw."""
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    _, va, te = _blocks(df, mission, [col])
    y, r = lab(te), te[col].to_numpy()
    days = te["g5_day"].to_numpy()
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([idx_by_day[d] for d in rng.choice(uniq, len(uniq))])
             for _ in range(n_boot)]
    rows, per_draw = [], {}
    for name, pol in fitted.items():
        s_va, s_te = pol.score(va), pol.score(te)
        ths = {d: threshold_for(s_va, d) for d in budgets}
        stats = {"auc": [], "ap": [], **{f"capture@{int(d * 100)}%": [] for d in budgets}}
        for ix in draws:
            yy, ss = y[ix], s_te[ix]
            stats["auc"].append(roc_auc_score(yy, ss))
            stats["ap"].append(average_precision_score(yy, ss))
            for d in budgets:
                stats[f"capture@{int(d * 100)}%"].append((ss[yy] >= ths[d]).mean())
        per_draw[name] = {k: np.asarray(v) for k, v in stats.items()}
        for k, v in stats.items():
            rows.append({"policy": name, "metric": k, "mean": np.mean(v),
                         "ci_low": np.quantile(v, 0.025), "ci_high": np.quantile(v, 0.975),
                         "n_boot": len(v)})
    best = max(per_draw, key=lambda p: per_draw[p]["auc"].mean())
    paired = []
    for other in per_draw:
        if other == best:
            continue
        for k in per_draw[best]:
            diff = per_draw[best][k] - per_draw[other][k]
            paired.append({"best": best, "other": other, "metric": k, "mean_diff": diff.mean(),
                           "ci_low": np.quantile(diff, 0.025), "ci_high": np.quantile(diff, 0.975),
                           "share_best_ahead": (diff > 0).mean(), "n_boot": len(diff)})
    return pd.DataFrame(rows), pd.DataFrame(paired)


# ---- E11: interpretation ---------------------------------------------------------------------

def e11_importance(df, mission, fitted, n_sub=50_000, seed=SEED):
    col = mission.label_col
    lab = label_fn(col, mission.threshold)
    _, _, te = _blocks(df, mission, [col])
    y = lab(te)
    learned = {k: v for k, v in fitted.items() if isinstance(v, LearnedPolicy)
               and set(v.features) == set(mission.tiers["A"])}
    best = max(learned, key=lambda k: roc_auc_score(y, learned[k].score(te)))
    pol = learned[best]
    sub = te.sample(min(n_sub, len(te)), random_state=seed)
    pi = permutation_importance(pol.model, sub[pol.features], lab(sub), scoring="roc_auc",
                                n_repeats=5, random_state=seed)
    perm = pd.DataFrame({"feature": pol.features, "auc_drop_mean": pi.importances_mean,
                         "auc_drop_std": pi.importances_std}).sort_values("auc_drop_mean",
                                                                         ascending=False)
    single = []
    for f in sorted(set(mission.tiers["A"]) | {"solar_hour"}):
        ok = te[f].notna().to_numpy()
        a = roc_auc_score(y[ok], te[f].to_numpy()[ok])
        single.append({"feature": f, "auc": max(a, 1 - a), "inverted": a < 0.5})
    return best, perm, pd.DataFrame(single).sort_values("auc", ascending=False)
