"""Data loading, splits, policies, the budget threshold rule, and metrics.

Definitions follow docs/baseline_paper/05_baselines.md: a policy produces a score per decision; the
gate fires on scores at or above the (1 - d) quantile of the *validation* scores (§5.4), and that
threshold is applied unchanged to the test block.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

G5NR_T0 = pd.Timestamp("2005-05-15 21:30")     # G5NR inst30mn axis, step 0
BUDGETS = (0.05, 0.10, 0.13, 0.20, 0.30, 0.50)
SEED = 0


# ---- data ------------------------------------------------------------------------------------

def load(path, mission):
    """Read an enriched dataset and add the columns every experiment needs."""
    df = pd.read_parquet(path)
    df["g5_time"] = G5NR_T0 + pd.to_timedelta(df["g5_step"] * 30, unit="min")
    df["g5_month"] = df["g5_time"].dt.strftime("%Y-%m")
    df["g5_day"] = df["g5_time"].dt.floor("D")
    sh = 2 * np.pi * df["solar_hour"] / 24
    df["sin_sh"], df["cos_sh"] = np.sin(sh), np.cos(sh)
    df["is_day"] = (df["solar_hour"] >= 6) & (df["solar_hour"] < 18)
    return df.sort_values("decision").reset_index(drop=True)


def valid(df, col):
    """Rows where target `col` is usable: validity flag set (if present) and finite."""
    ok = df[col].notna() & np.isfinite(df[col])
    if f"{col}_valid" in df:
        ok &= df[f"{col}_valid"].astype(bool)
    return ok


def blocked_split(df, mission):
    """Train/val/test by G5NR month, dropping one weather step on each side of every boundary."""
    part = pd.Series("drop", index=df.index)
    for name, months in (("train", mission.train_months), ("val", mission.val_months),
                         ("test", mission.test_months)):
        part[df["g5_month"].isin(months)] = name
    steps = df["g5_step"]
    for a, b in (("train", "val"), ("val", "test")):
        last_a, first_b = steps[part == a].max(), steps[part == b].min()
        part[steps.isin([last_a, first_b])] = "drop"
    return part


def random_split(df, frac_test=0.3, frac_val=0.1, seed=SEED):
    """Row-level random split (the leaky comparison in E1)."""
    u = np.random.default_rng(seed).random(len(df))
    return pd.Series(np.where(u < frac_test, "test", np.where(u < frac_test + frac_val, "val",
                                                                 "train")), index=df.index)


# ---- policies --------------------------------------------------------------------------------

def _lr(features):
    pre = ColumnTransformer([("x", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler())]), features)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])


def make_model(kind, features, seed=SEED):
    """Fitted-model factory for learned policies. Every model exposes predict_proba on a frame."""
    if kind == "lr":
        return _lr(features)
    if kind == "lgbm":                  # NaN handled natively
        return Pipeline([("sel", ColumnTransformer([("x", "passthrough", features)],
                                                   verbose_feature_names_out=False)),
                         ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                 num_leaves=31, subsample=0.8, subsample_freq=1,
                                                 colsample_bytree=0.9, random_state=seed,
                                                 n_jobs=-1, verbose=-1))])
    if kind == "mlp":
        pre = ColumnTransformer([("x", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())]), features)])
        return Pipeline([("pre", pre), ("mlp", MLPClassifier((64, 64), max_iter=200,
                                                             early_stopping=True,
                                                             random_state=seed))])
    raise ValueError(kind)


class LearnedPolicy:
    def __init__(self, kind, features, seed=SEED):
        self.kind, self.features = kind, list(features)
        self.model = make_model(kind, self.features, seed)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def score(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]


class RandomPolicy:
    kind = "random"

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        return self

    def score(self, X):
        return self.rng.random(len(X))


class LatitudeBandPolicy:
    """P3: score = training positive rate of the decision's latitude band (5° bands)."""
    kind = "lat_band"

    def __init__(self, width=5.0, seed=SEED):
        self.width, self.rng = width, np.random.default_rng(seed)

    def _band(self, X):
        return np.floor(X["lat"].to_numpy() / self.width)

    def fit(self, X, y):
        self.rate = pd.Series(np.asarray(y, float)).groupby(self._band(X)).mean().to_dict()
        return self

    def score(self, X):
        s = np.array([self.rate[b] for b in self._band(X)])
        return s + 1e-9 * self.rng.random(len(s))      # break ties so the budget is met exactly


class SingleFeaturePolicy:
    """P4: tautot_mean by day, -cldtmp_mean by night, each mapped to its training ECDF so the two
    regimes share one score scale."""
    kind = "threshold"

    def fit(self, X, y):
        self.day_ref = np.sort(X.loc[X["is_day"], "tautot_mean"].dropna().to_numpy())
        self.night_ref = np.sort(-X.loc[~X["is_day"], "cldtmp_mean"].dropna().to_numpy())
        return self

    @staticmethod
    def _ecdf(ref, v):
        out = np.searchsorted(ref, v, side="right") / max(len(ref), 1)
        return np.where(np.isnan(v), -1.0, out)

    def score(self, X):
        day = X["is_day"].to_numpy()
        s = np.empty(len(X))
        s[day] = self._ecdf(self.day_ref, X.loc[day, "tautot_mean"].to_numpy())
        s[~day] = self._ecdf(self.night_ref, -X.loc[~day, "cldtmp_mean"].to_numpy())
        return s


class SwitchedPolicy:
    """Day/night switch: one learned policy for day decisions, another for night (E5)."""

    def __init__(self, day_policy, night_policy):
        self.day, self.night = day_policy, night_policy

    def fit(self, X, y):
        y = np.asarray(y)
        d = X["is_day"].to_numpy()
        self.day.fit(X[d], y[d])
        self.night.fit(X[~d], y[~d])
        return self

    def score(self, X):
        d = X["is_day"].to_numpy()
        s = np.empty(len(X))
        s[d] = self.day.score(X[d])
        s[~d] = self.night.score(X[~d])
        return s


# ---- budget rule and metrics -----------------------------------------------------------------

def threshold_for(scores_val, d):
    """Score threshold that fires on a fraction d of validation decisions."""
    return np.quantile(scores_val, 1 - d)


def gate_metrics(fire, y, r, scores=None):
    """capture@d, mass-capture@d, precision@d, lift@d, realized duty, (AUC, AP)."""
    fire, y = np.asarray(fire, bool), np.asarray(y, bool)
    r = np.clip(np.nan_to_num(np.asarray(r, float)), 0, None)
    out = {"duty": fire.mean(), "capture": fire[y].mean(),
           "mass_capture": r[fire].sum() / r.sum(), "precision": y[fire].mean(),
           "base_rate": y.mean()}
    out["lift"] = out["capture"] / out["duty"]
    if scores is not None:
        out["auc"] = roc_auc_score(y, scores)
        out["ap"] = average_precision_score(y, scores)
    return out


def evaluate(policy, train, val, test, label, mass_col, budgets=BUDGETS):
    """Fit on train, set each budget's threshold on val, score test; mass-capture is weighted by
    `mass_col`. Returns (rows, test scores)."""
    policy.fit(train, label(train))
    s_val, s_test = policy.score(val), policy.score(test)
    y, r = label(test), test[mass_col].to_numpy()
    rows = []
    for d in budgets:
        th = threshold_for(s_val, d)
        rows.append({"d": d, **gate_metrics(s_test >= th, y, r, s_test)})
    return rows, s_test


def oracle_rows(test, label, rank_col, mass_col, budgets=BUDGETS):
    """P8: fires on the highest true label rates (`rank_col`) first, so positives come before
    negatives; mass-capture is weighted by `mass_col`."""
    y, r = label(test), np.nan_to_num(test[rank_col].to_numpy())
    w = test[mass_col].to_numpy()
    order = np.argsort(-r, kind="stable")
    rows = []
    for d in budgets:
        fire = np.zeros(len(r), bool)
        fire[order[:int(round(d * len(r)))]] = True
        rows.append({"d": d, **gate_metrics(fire, y, w, r)})
    return rows
