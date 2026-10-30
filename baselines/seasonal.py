"""Seasonal LOMO experiments. All fitting/calibration uses training-side rows only.

The chronological experiments remain in experiments.py and their results are never rewritten.
E2/E3 share fitted policies; E6/E10/E11 reuse those predictions. P8 is an unattainable reference.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import (BUDGETS, SEED, LearnedPolicy, SwitchedPolicy, fit_seasonal_policy,
                   gate_metrics, seasonal_folds, threshold_for, valid)
from .experiments import label_fn, policies

PROTOCOL = "seasonal_lomo_v1"
THRESHOLD_SOURCE = "training_months_in_sample_score_quantile"
CURVE = tuple(np.round(np.linspace(.01, 1, 100), 2))
METRICS = ["auc", "average_precision", "capture", "mass_capture", "precision",
           "nominal_budget", "realized_duty", "base_rate"]


def ranking(y, scores):
    y = np.asarray(y, bool)
    return {"auc": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
            "average_precision": average_precision_score(y, scores) if y.any() else np.nan}


def metadata(mission, fold, train, test, dataset_hash, seed=SEED):
    return {"mission": mission.name, "held_out_month": fold.held_out_month,
            "train_months": json.dumps(fold.train_months), "n_train": len(train), "n_test": len(test),
            "seed": seed, "dataset_hash": dataset_hash, "protocol": PROTOCOL,
            "threshold_source": THRESHOLD_SOURCE,
            "buffer_steps": json.dumps(fold.buffer_steps)}


def scored_rows(s_train, s_test, y, mass, budgets=BUDGETS, *, oracle=False):
    """Ranking computed once; every budget records the fixed threshold and realized duty."""
    rank = ranking(y, s_test)
    rows = []
    for budget in budgets:
        if oracle:
            fire = np.zeros(len(y), bool)
            fire[np.argsort(-s_test, kind="stable")[:round(budget * len(y))]] = True
            threshold = np.nan
        else:
            threshold = threshold_for(s_train, budget)
            fire = s_test >= threshold
        g = gate_metrics(fire, y, mass)
        rows.append({**g, **rank, "nominal_budget": budget, "realized_duty": g["duty"],
                     "score_threshold": threshold,
                     "threshold_source": "oracle_held_out_truth_reference" if oracle else THRESHOLD_SOURCE})
    return rows


def evaluate_variant(mission, fold, train, test, estimator, lab, mass_col, dataset_hash,
                     budgets=BUDGETS, **tags):
    meta = metadata(mission, fold, train, test, dataset_hash)
    if not len(test) or not len(train) or len(np.unique(lab(train))) < 2:
        return [{**meta, **tags, "status": "empty_or_single_class_training",
                 "nominal_budget": d, "base_rate": np.mean(lab(test)) if len(test) else np.nan}
                for d in budgets]
    _, st, se = fit_seasonal_policy(estimator, train, test, lab)
    return [{**meta, **tags, "status": "ok", **r}
            for r in scored_rows(st, se, lab(test), test[mass_col], budgets)]


def summarize(frame, groups, metrics=METRICS):
    """Equal-weight month summaries, never pooled row metrics. SD uses ddof=1."""
    available = [m for m in metrics if m in frame and m not in groups]
    out = frame.groupby(groups, dropna=False)[available].agg(["mean", "std", "min", "max", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def missing_rows(df, mission, fold, dataset_hash):
    tr, te = fold.frames(df)
    meta = metadata(mission, fold, tr, te, dataset_hash)
    rows = []
    for part, frame in (("train", tr), ("held_out", te), ("buffer", df.loc[fold.dropped_index])):
        for feat in mission.tiers["A"]:
            missing = int(frame[feat].isna().sum())
            rows.append({**meta, "partition": part, "item": feat, "n_rows": len(frame),
                         "missing_count": missing, "available_count": len(frame)-missing,
                         "availability": 1-missing/len(frame) if len(frame) else np.nan,
                         "training_missing_indicator": bool(tr[feat].isna().any()),
                         "note": "LR/MLP imputer/scaler/indicators fitted on train; LightGBM native NaN"})
        for col in sorted({mission.label_col, mission.swath_col, mission.mass_col, mission.convective_col}):
            count = int((~valid(frame, col)).sum())
            rows.append({**meta, "partition": part, "item": f"invalid target: {col}",
                         "n_rows": len(frame), "missing_count": count,
                         "available_count": len(frame)-count})
    return rows


def _weighted_ranking(y, scores, day_index, weights):
    """Exact day-weight bootstrap AUC/AP, sorting once and handling score ties in groups.

    A draw's day multiplicities equal physically repeating each sampled day's rows. Batches
    bound memory; this avoids thousands of repeated sorts/concatenations per policy and fold.
    """
    order = np.argsort(-scores, kind="stable")
    yy, ss, di = y[order], scores[order], day_index[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    auc, ap = [], []
    for offset in range(0, len(weights), 16):
        w = weights[offset:offset+16, di].astype(float)
        tp = np.cumsum(np.add.reduceat(w * yy, starts, axis=1), axis=1)
        fp = np.cumsum(np.add.reduceat(w * ~yy, starts, axis=1), axis=1)
        positives, negatives = tp[:, -1], fp[:, -1]
        dtp = np.diff(tp, prepend=0, axis=1)
        dfp = np.diff(fp, prepend=0, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            auc.extend(np.sum((tp - .5*dtp)*dfp, axis=1)/(positives*negatives))
            precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
            ap.extend(np.sum(dtp * precision, axis=1)/positives)
    return np.asarray(auc), np.asarray(ap)


def bootstrap_month(test, y, mass, cache, n_boot=1000, seed=SEED):
    days, di = np.unique(test.g5_day.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(days), np.full(len(days), 1/len(days)), size=n_boot)
    def sums(v):
        return weights @ np.bincount(di, weights=np.asarray(v, float), minlength=len(days))
    den_y, den_mass, den_n = sums(y), sums(mass), sums(np.ones(len(y)))
    records, draws = [], {}
    for name, (_, st, se) in cache.items():
        auc, ap = _weighted_ranking(y, se, di, weights)
        metrics = {"auc": auc, "average_precision": ap}
        for budget in (.13, .20):
            if name == "P8 oracle":
                fire = np.zeros(len(y), bool)
                fire[np.argsort(-se, kind="stable")[:round(budget*len(y))]] = True
            else:
                fire = se >= threshold_for(st, budget)
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics.update({f"capture@{budget:.0%}": sums(fire & y)/den_y,
                                f"mass_capture@{budget:.0%}": sums(fire*mass)/den_mass,
                                f"precision@{budget:.0%}": sums(fire & y)/sums(fire),
                                f"realized_duty@{budget:.0%}": sums(fire)/den_n})
        draws[name] = metrics
        for metric, v in metrics.items():
            finite = v[np.isfinite(v)]
            records.append({"policy": name, "threshold_source": "oracle_held_out_truth_reference" if name == "P8 oracle" else THRESHOLD_SOURCE, "metric": metric, "mean": np.mean(finite) if len(finite) else np.nan,
                            "ci_low": np.quantile(finite,.025) if len(finite) else np.nan,
                            "ci_high": np.quantile(finite,.975) if len(finite) else np.nan,
                            "n_boot": n_boot, "n_boot_valid": len(finite), "n_days": len(days),
                            "resampling": "whole G5NR days within held-out month; fixed gate"})
    # Fixed, prespecified reference avoids selecting the best model using held-out outcomes.
    paired = []
    ref = "P6 gradient boosting"
    for other in draws:
        if other in (ref, "P8 oracle"):
            continue
        for metric in draws[ref]:
            v = draws[ref][metric]-draws[other][metric]
            paired.append({"reference": ref, "other": other, "metric": metric,
                           "mean_diff": np.nanmean(v), "ci_low": np.nanquantile(v,.025),
                           "ci_high": np.nanquantile(v,.975), "n_boot": n_boot})
    return records, paired


def run_fold(df, mission, fold, dataset_hash, *, include_mlp=True, n_boot=1000,
             importance_rows=50_000, importance_repeats=5, log=print):
    """Run E2-E8/E10-E11 on a single central fold; E1/E9 retain historical results."""
    col = mission.label_col
    tr, te = fold.frames(df, [col, mission.mass_col])
    lab = label_fn(col, mission.threshold)
    meta = metadata(mission, fold, tr, te, dataset_hash)
    cache, curves, e2 = {}, [], []
    for name, pol in policies(mission, include_mlp).items():
        log(f"{fold.held_out_month} {name}")
        cache[name] = fit_seasonal_policy(pol, tr, te, lab)
    cache["P8 oracle"] = (None, None, te[col].to_numpy())
    for name, (_, st, se) in cache.items():
        common = {**meta, "policy": name, "reference_only": name == "P8 oracle"}
        rows = scored_rows(st, se, lab(te), te[mission.mass_col], CURVE, oracle=name == "P8 oracle")
        curves += [{**common, **r} for r in rows]
        e2 += [{**common, **r, "budget_rule": "fixed"} for r in rows
               if any(np.isclose(r["nominal_budget"], d) for d in BUDGETS)]
        base = float(lab(tr).mean())
        e2 += [{**common, **r, "budget_rule": "training_base_rate"}
               for r in scored_rows(st, se, lab(te), te[mission.mass_col], (base,), oracle=name == "P8 oracle")]
    outputs = {"e2_monthly": pd.DataFrame(e2), "e3_capture_curves": pd.DataFrame(curves)}

    log(f"{fold.held_out_month} E4 label geometry")
    geometry = []
    for variable, columns in mission.label_geometries.items():
        train, test = fold.frames(df, columns)
        common = metadata(mission, fold, train, test, dataset_hash)
        for train_on in columns:
            pol, st, se = fit_seasonal_policy(LearnedPolicy("lgbm", mission.tiers["A"]), train, test,
                                             label_fn(train_on, mission.threshold))
            for score_on in columns:
                geometry += [{**common, "policy": "P6 gradient boosting", "variable": variable,
                              "trained_on": train_on, "scored_on": score_on,
                              "mass_column": score_on, **r}
                             for r in scored_rows(st, se, label_fn(score_on, mission.threshold)(test),
                                                  test[score_on], (.20,))]
    outputs["e4_label_geometry"] = pd.DataFrame(geometry)

    log(f"{fold.held_out_month} E5 tiers and illumination")
    tiers = []
    for tier, feats in mission.tiers.items():
        for regime, is_day in (("day", True), ("night", False)):
            if regime == "night" and any(f.startswith("tautot") for f in feats):
                tiers.append({**meta, "policy": "P6 gradient boosting", "tier": tier, "regime": regime,
                              "nominal_budget": .20, "status": "not_available_optical_depth_at_night"})
                continue
            tiers += evaluate_variant(mission, fold, tr[tr.is_day == is_day], te[te.is_day == is_day],
                                      LearnedPolicy("lgbm", feats), lab, mission.mass_col, dataset_hash,
                                      CURVE, policy="P6 gradient boosting", tier=tier, regime=regime)
        switched = SwitchedPolicy(LearnedPolicy("lgbm", feats),
                                  LearnedPolicy("lgbm", [f for f in feats if not f.startswith("tautot")]))
        tiers += evaluate_variant(mission, fold, tr, te, switched, lab, mission.mass_col, dataset_hash,
                                  BUDGETS, policy="P6 gradient boosting", tier=tier, regime="switched")
    outputs["e5_tiers"] = pd.DataFrame(tiers)

    _, st, se = cache["P6 gradient boosting"]
    alat = te.lat.abs()
    strata = {name: ((alat >= lo) & (alat < hi)).to_numpy() for name, lo, hi in mission.lat_strata}
    strata.update(ocean=(te.ocean_frac_nadir > .95).to_numpy(), land=(te.land_frac_nadir > .95).to_numpy(),
                  mixed=((te.ocean_frac_nadir <= .95) & (te.land_frac_nadir <= .95)).to_numpy())
    rows = []
    for name, mask in strata.items():
        rows += [{**meta, "policy": "P6 gradient boosting", "stratum": name,
                  "n_test": int(mask.sum()), "threshold_scope": "global_training_fold", **r}
                 for r in scored_rows(st, se[mask], lab(te)[mask], te[mission.mass_col].to_numpy()[mask])]
    outputs["e6_strata"] = pd.DataFrame(rows)

    log(f"{fold.held_out_month} E7 label sensitivity")
    variants = [(col, mission.threshold*x, False, f"rain_{x:g}_mm_hr") for x in (.1,1,5)]
    variants += [(mission.convective_col, mission.threshold, False, "convective_1_mm_hr"),
                 (col, float(tr[col].median()), True, "training_median_split")]
    labels = []
    for target, threshold, strict, name in variants:
        train, test = fold.frames(df, [target])
        label = label_fn(target, threshold, strict)
        base = float(label(train).mean())
        for kind, pname in (("lr", "P5 logistic regression"), ("lgbm", "P6 gradient boosting")):
            labels += evaluate_variant(mission, fold, train, test, LearnedPolicy(kind,mission.tiers["A"]),
                                       label, target, dataset_hash, (*BUDGETS, base),
                                       policy=pname, label=name, label_column=target,
                                       label_threshold=threshold, training_base_rate=base,
                                       label_threshold_source="training_median" if strict else "prespecified_physical_threshold")
    outputs["e7_label_sensitivity"] = pd.DataFrame(labels)
    outputs["e8_missing_data"] = pd.DataFrame(missing_rows(df, mission, fold, dataset_hash))

    log(f"{fold.held_out_month} E10 {n_boot} within-month day bootstraps")
    boot, paired = bootstrap_month(te, lab(te), te[mission.mass_col].to_numpy(), cache, n_boot=n_boot)
    outputs["e10_bootstrap"] = pd.DataFrame([{**meta, **r} for r in boot])
    outputs["e10_bootstrap_paired"] = pd.DataFrame([{**meta, **r} for r in paired])

    log(f"{fold.held_out_month} E11 importance")
    # P6 prespecified for interpretation; no held-out selection of the 'best' estimator.
    pol = cache["P6 gradient boosting"][0]
    sub = te.sample(min(importance_rows, len(te)), random_state=SEED)
    pi = permutation_importance(pol.model, sub[pol.features], lab(sub), scoring="roc_auc",
                                n_repeats=importance_repeats, random_state=SEED)
    outputs["e11_permutation_importance"] = pd.DataFrame([
        {**meta, "policy": "P6 gradient boosting", "feature": f, "auc_drop_mean": mean,
         "auc_drop_std": std, "n_importance": len(sub), "n_repeats": importance_repeats}
        for f, mean, std in zip(pol.features, pi.importances_mean, pi.importances_std)])
    single = []
    for feat in sorted(set(mission.tiers["A"]) | {"solar_hour"}):
        train_ok, test_ok = np.isfinite(tr[feat]), np.isfinite(te[feat])
        a = ranking(lab(tr)[train_ok], tr.loc[train_ok,feat].to_numpy())["auc"]
        sign = -1 if a < .5 else 1
        single.append({**meta, "feature": feat, "direction_source": "training_months_auc",
                       "sign": sign, "n_available_test": int(test_ok.sum()),
                       **ranking(lab(te)[test_ok], sign*te.loc[test_ok,feat].to_numpy())})
    outputs["e11_single_feature_auc"] = pd.DataFrame(single)
    return outputs


def aggregate(out):
    """Aggregate completed fold files; bootstrap bounds are descriptive across folds, not a CI
    on the mean of correlated cross-validation folds."""
    out = Path(out)
    tables = sorted({p.name for p in (out/"folds").glob("*/*.csv")})
    for table in tables:
        paths = sorted((out/"folds").glob(f"*/{table}"))
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        frame.to_csv(out/table,index=False)
    e2 = pd.read_csv(out/"e2_monthly.csv")
    summarize(e2[e2.budget_rule == "fixed"], ["mission","policy","nominal_budget"]).to_csv(
        out/"e2_summary.csv",index=False)
    configs = {
        "e4_label_geometry": (["mission","policy","variable","trained_on","scored_on"], METRICS),
        "e5_tiers": (["mission","policy","tier","regime","nominal_budget"], METRICS),
        "e6_strata": (["mission","policy","stratum","nominal_budget"], METRICS),
        "e7_label_sensitivity": (["mission","policy","label"], METRICS),
        "e10_bootstrap": (["mission","policy","metric"], ["mean","ci_low","ci_high"]),
        "e11_permutation_importance": (["mission","policy","feature"], ["auc_drop_mean"]),
        "e11_single_feature_auc": (["mission","feature"], ["auc","average_precision"]),
    }
    for name,(groups,metrics) in configs.items():
        frame = pd.read_csv(out/f"{name}.csv")
        if name == "e7_label_sensitivity":
            # Avoid averaging across different budgets within a month.
            frame = frame[np.isclose(frame.nominal_budget,.20)]
        summarize(frame,groups,metrics).to_csv(out/f"{name}_summary.csv",index=False)
