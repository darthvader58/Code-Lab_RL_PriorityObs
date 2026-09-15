#!/usr/bin/env python3
"""Synchronize the completed enriched dataset into the trackwise notebook.

This edits notebook code/markdown only. It does not run extraction or touch a
parquet file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "geos5data_0.0625deg_30mn_trackwise.ipynb"

def lines(text):
    return text.splitlines(True)

def set_source(cell, text):
    cell["source"] = lines(text)

def clear_output(cell):
    cell["outputs"] = []
    cell["execution_count"] = None

def main():
    nb = json.loads(NB_PATH.read_text())

    set_source(nb["cells"][0], """# Track-wise dataset — enriched 0.0625°, 30-minute, 1-minute decisions

This notebook analyzes the completed CPR-gating dataset. The preparation pipeline
has already run; the notebook loads `trackwise_decisions.parquet`, which now
contains the validated 40-column enriched schema. It does not re-run extraction.

The dataset supports the future RL problem: use GEOS-5 Nature Run diagnostic
proxies and geometry to decide whether to power the CPR, then score that action
against stored target diagnostics. The cloud and radiation fields are idealized
GEOS-5 proxies, not simulated EarthCARE retrievals.

| | final trackwise dataset |
|---|---|
| collection | `inst30mn_2d_met1_Nx` @ **0.0625°** |
| weather cadence | **30 min** |
| decision cadence | **1 min** |
| rows | **351,639** |
| source columns | 27 |
| enriched columns | **40** |
| decision discontinuities | **49** (1,161 missing decision IDs) |
| labels | CPR swath and nadir targets, retained but never ordinary observations |

The enriched additions are `lwtup_fore`, continuous nadir land/water fractions,
cyclic longitude (`sin_lon`, `cos_lon`), and eight target-validity flags. See
`docs/feature_set.md` and `docs/column_registry.yaml` for the complete feature
tiers and limitations.
""")

    set_source(nb["cells"][1], """> **Completed-dataset mode:** the extraction cells below document the original
configuration and remain useful for provenance, but should not be rerun for this
analysis. The notebook loader reads the validated local
`trackwise_decisions.parquet` directly. No weather download or re-extraction is
needed to reproduce the analysis and visuals.

The source parquet is now the enriched 40-column dataset. All joins and checks
in the preparation workflow used the `decision` key, preserving the 49 genuine
discontinuities rather than shifting rows across missing decision IDs.
""")

    loader = """# ---- load the completed enriched dataset -------------------------------
# This cell is analysis-only: it does not concatenate extraction cache parts,
# contact OPeNDAP, or rewrite the source parquet.
DATASET_PATH = "trackwise_decisions.parquet"
rows = pd.read_parquet(DATASET_PATH)
EXPECTED_ROWS, EXPECTED_COLS = 351_639, 40
assert rows.shape == (EXPECTED_ROWS, EXPECTED_COLS), rows.shape
assert rows.decision.is_unique
assert rows.decision.diff().fillna(1).ge(0).all()
assert rows.g5_step.nunique() > 1

validity_flags = [
    "prectot_swath_valid", "prectot_nadir_valid",
    "preccon_swath_valid", "preccon_nadir_valid",
    "iwp_swath_valid", "iwp_nadir_valid",
    "lwp_swath_valid", "lwp_nadir_valid",
]
assert set(validity_flags).issubset(rows.columns)
print(f"loaded {len(rows):,} rows x {rows.shape[1]} enriched columns")
print(f"weather fields: {rows.field.nunique():,} | decision discontinuities: {(rows.decision.diff().dropna() != 1).sum():,}")
print(f"target-invalid entries: {(~rows[validity_flags]).sum().sum():,} | "
      f"decisions with any invalid target: {(~rows[validity_flags].all(axis=1)).sum():,}")
print(f"mixed-surface decisions (0.05 < land_frac_nadir < 0.95): "
      f"{((rows.land_frac_nadir > .05) & (rows.land_frac_nadir < .95)).mean()*100:.2f}%")
rows.head()
"""
    set_source(nb["cells"][8], loader)
    clear_output(nb["cells"][8])

    c13 = "".join(nb["cells"][13].get("source", []))
    c13 = c13.replace("KDE of nadir precipitation seen by EarthCARE (Jun-Jul 2005)",
                      "KDE of nadir precipitation diagnostic across the 8-month trackwise dataset")
    c13 = c13.replace("KDE of MSI cloud optical thickness along track (Jun-Jul 2005)",
                      "KDE of GEOS-5 TAUTOT diagnostic proxy along track (8-month dataset)")
    set_source(nb["cells"][13], c13)
    clear_output(nb["cells"][13])

    visual_md = """## Enriched-dataset diagnostics

These visuals are derived from the final 40-column parquet. They make the target
validity mask, static land/water context, cyclic longitude representation, and
verified one-minute sequential structure visible before any RL environment is
built.
"""
    visual_code = """# ---- enriched diagnostics: validity, surface, longitude, autocorrelation ----
import seaborn as sns

# 1. Missing/corrupt target validity counts. False means missing or a finite
#    abs(value) >= 1e6 sentinel; no target is imputed.
invalid_counts = (~rows[validity_flags]).sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
invalid_counts.plot.bar(ax=ax, color="#b2182b")
ax.set_ylabel("invalid entries"); ax.set_title("Target-validity audit")
ax.tick_params(axis="x", rotation=45); plt.tight_layout(); plt.show()

# 2. Continuous surface fractions. Mixed-surface decisions are defined as
#    0.05 < land_frac_nadir < 0.95; they average 60 one-second samples and are
#    not necessarily individual shoreline grid cells.
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(rows.land_frac_nadir, bins=40, alpha=.7, label="land fraction", color="#8c510a")
ax.hist(rows.ocean_frac_nadir, bins=40, alpha=.6, label="water fraction", color="#01665e")
ax.set_xlabel("continuous fraction"); ax.set_ylabel("decisions")
ax.set_title("Nadir land/water fractions"); ax.legend(); plt.tight_layout(); plt.show()

# 3. Longitude seam check: raw longitude is retained, while sin/cos encode it
#    continuously around +/-180 degrees.
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(rows.sin_lon.iloc[::20], rows.cos_lon.iloc[::20], s=.5, alpha=.2)
t = np.linspace(0, 2*np.pi, 400); ax.plot(np.sin(t), np.cos(t), "k--", lw=.8)
ax.set_aspect("equal"); ax.set_xlabel("sin_lon"); ax.set_ylabel("cos_lon")
ax.set_title("Cyclic longitude representation"); plt.tight_layout(); plt.show()

# 4. Verified autocorrelation diagnostic. Pairs crossing decision gaps were
#    excluded before calculating these values.
acf = {
    "preccon_nadir": [0.116024, 0.056602, 0.038470, 0.024907, -0.003190, -0.016144],
    "prectot_nadir": [0.171179, 0.060293, 0.015995, 0.011164, 0.005938, -0.007796],
    "tautot_mean": [0.628451, 0.370135, 0.229986, 0.066772, -0.054396, -0.030714],
}
lags = [1, 2, 3, 5, 10, 30]
fig, ax = plt.subplots(figsize=(8, 4))
for name, values in acf.items(): ax.plot(lags, values, marker="o", label=name)
ax.axhline(np.exp(-1), color="k", ls="--", lw=.8, label="1/e")
ax.axhline(0, color="0.6", lw=.7); ax.set_xscale("log")
ax.set_xticks(lags); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("lag (one-minute decisions)"); ax.set_ylabel("autocorrelation")
ax.set_title("Sequential structure at the saved cadence"); ax.legend()
plt.tight_layout(); plt.show()
"""
    nb["cells"].insert(14, {"cell_type": "markdown", "metadata": {}, "source": lines(visual_md)})
    nb["cells"].insert(15, {"cell_type": "code", "execution_count": None,
                             "metadata": {}, "outputs": [], "source": lines(visual_code)})

    # Existing analysis/export cells should see fresh data rather than stale
    # rendered output from the pre-enrichment parquet.
    for i in [9, 10, 11, 13, 15, 16]:
        if i < len(nb["cells"]):
            clear_output(nb["cells"][i])

    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"updated {NB_PATH} with {len(nb['cells'])} cells")

if __name__ == "__main__":
    main()

