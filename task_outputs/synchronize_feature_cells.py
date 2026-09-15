import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "geos5data_0.0625deg_30mn_trackwise.ipynb"
nb = json.loads(p.read_text())

def set_src(i, text):
    nb["cells"][i]["source"] = text.splitlines(True)
    if nb["cells"][i].get("cell_type") == "code":
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None

set_src(9, """# ---- CLASS BALANCE: valid swath vs nadir targets -----------------------
MM = 3600.0     # kg m-2 s-1 -> mm/hr
print(f"{len(rows):,} decisions ({DECISION.total_seconds()/60:.0f}-minute interval)\\n")
print(f"{'threshold':<20}{'prectot swath':>20}{'prectot nadir':>20}"
      f"{'preccon swath':>20}{'preccon nadir':>20}")
for nm, mm in [("any > 0", 0.0), (">= 0.1 mm/hr", 0.1), (">= 0.5 mm/hr", 0.5),
               (">= 1 mm/hr", 1.0), (">= 5 mm/hr", 5.0), (">= 10 mm/hr", 10.0)]:
    t = mm / MM
    cells = []
    for c in ["prectot_swath", "prectot_nadir", "preccon_swath", "preccon_nadir"]:
        valid = rows[c + "_valid"]
        p = 100 * (rows.loc[valid, c] > t).mean()
        cells.append(f"{int((rows.loc[valid, c] > t).sum()):>7,} ({p:4.1f}%, n={int(valid.sum()):,})")
    print(f"  {nm:<18}" + "".join(f"{c:>20}" for c in cells))

print("\\nThe nadir column is what the CPR would actually measure; the swath column")
print("credits it with precipitation it never flies over. Validity flags exclude")
print("missing/corrupt targets from every positive-rate calculation.")
for c in ["prectot_swath", "prectot_nadir"]:
    valid = rows[c + "_valid"]
    r = (rows.loc[valid, c] > 1.0 / MM).mean()
    print(f"  {c:<15} at >=1 mm/hr: {100*r:.1f}% positive among {int(valid.sum()):,} valid targets")
""")

set_src(10, """# ---- learnability: always-available features vs valid targets ------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# These are the registry's always-available candidates. CPR-conditional
# diagnostics and target columns are deliberately excluded.
FEATS = [
    "cldtot_mean", "cldtmp_mean", "tautot_mean", "tautot_max",
    "lwtup_mean", "lwtup_fore", "solar_hour", "lat", "sin_lon", "cos_lon",
    "land_frac_nadir", "ocean_frac_nadir",
]

def auc(score, y):
    m = np.isfinite(score)
    return roc_auc_score(y[m], score[m]) if len(set(y[m])) == 2 else float("nan")

for lab in ["prectot_nadir", "prectot_swath"]:
    valid = rows[lab + "_valid"].to_numpy()
    y = (rows.loc[valid, lab] > 1.0 / MM).astype(int).to_numpy()
    print(f"\\n=== label: {lab} >= 1 mm/hr ({100*y.mean():.1f}% positive; {len(y):,} valid rows) ===")
    scored = []
    for f in FEATS:
        score = rows.loc[valid, f].to_numpy(dtype=float)
        scored.append((auc(score, y), f))
    sk = sorted(scored, key=lambda z: -abs((z[0] if np.isfinite(z[0]) else .5) - .5))
    for a, f in sk[:8]:
        print(f"  {f:<20}{a:6.3f}" + (f"   (inverted {1-a:.3f})" if a < 0.5 else ""))

valid = rows["prectot_nadir_valid"].to_numpy()
X = rows.loc[valid, FEATS].replace([np.inf, -np.inf], np.nan)
X = ((X - X.mean()) / X.std()).fillna(0).values
y = (rows.loc[valid, "prectot_nadir"] > 1.0 / MM).astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
print(f"\\nlogistic on {len(FEATS)} always-available features -> valid nadir prectot >= 1 mm/hr")
print(f"  ROC-AUC {roc_auc_score(yte,p):.3f} | avg-precision "
      f"{average_precision_score(yte,p):.3f} (base rate {yte.mean():.3f})")
""")

set_src(11, """# ---- valid-target distributions and class balance ----------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
MM = 3600.0
sw = rows.loc[rows.prectot_swath_valid, "prectot_swath"].clip(lower=1e-9)
na = rows.loc[rows.prectot_nadir_valid, "prectot_nadir"].clip(lower=1e-9)
ax[0].hist(np.log10(sw * MM), bins=60, alpha=.6, label="swath")
ax[0].hist(np.log10(na * MM), bins=60, alpha=.6, label="nadir")
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].set_xlabel("log10 peak prectot (mm/hr)"); ax[0].set_ylabel("valid decisions")
ax[0].set_title("Valid target distribution: swath vs nadir"); ax[0].legend()

th = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10])
for c, lb in [("prectot_swath", "swath"), ("prectot_nadir", "nadir")]:
    valid = rows[c + "_valid"]
    ax[1].plot(th, [100 * (rows.loc[valid, c] > t / MM).mean() for t in th],
                marker="o", label=lb)
ax[1].axhline(50, color="k", ls="--", lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel("threshold (mm/hr)")
ax[1].set_ylabel("% positive among valid targets")
ax[1].set_title("Valid-target class balance vs threshold"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

# Update target handling in the geographic visual/export cells.
for idx in [13, 16]:
    src = "".join(nb["cells"][idx].get("source", []))
    src = src.replace(
        'THR = rows.prectot_nadir.quantile(0.50)      # balanced label for THIS window',
        'THR = rows.loc[rows.prectot_nadir_valid, "prectot_nadir"].quantile(0.50)      # balanced valid-target label')
    src = src.replace(
        'rows["fire"] = (rows.prectot_nadir > THR).astype(int)',
        'rows["fire"] = np.nan\nrows.loc[rows.prectot_nadir_valid, "fire"] = (rows.loc[rows.prectot_nadir_valid, "prectot_nadir"] > THR).astype(int)')
    src = src.replace(
        'THR = rows.prectot_nadir.quantile(0.50)',
        'THR = rows.loc[rows.prectot_nadir_valid, "prectot_nadir"].quantile(0.50)')
    nb["cells"][idx]["source"] = src.splitlines(True)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

p.write_text(json.dumps(nb, indent=1) + "\n")
print("synchronized feature, target-validity, and visualization cells")
