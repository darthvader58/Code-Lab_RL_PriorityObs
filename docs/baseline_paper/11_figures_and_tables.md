# 11. Figures and Tables Plan

**Evaluation scope.** Seasonal leave-one-month-out is the primary seasonal robustness evaluation
for EarthCARE and TRMM, separately. See [07_experiments.md](07_experiments.md) for the common fold
protocol and [16_seasonal_results.md](16_seasonal_results.md) for generated monthly results.
The legacy figures and numbers below describe the **Chronological future-season stress test**
(June–September training, October calibration, November–January testing including the partial
February tail), except explicitly labeled legacy E2 LOMO results. A fixed later-season test does
not establish all-season reliability.


Aim for 6–7 figures and 5–6 tables in a full paper; a 4-page conference paper (e.g., IGARSS) keeps
Fig. 1, Fig. 3, Fig. 4, Table 2 and Table 3 only, for EarthCARE. Multi-panel figures get one panel per
mission (EarthCARE, TRMM, FY-3G) with identical axes. Status ◐ = EarthCARE panel exists, other
missions to come. EarthCARE figures are drawn by `python -m baselines.paper_figures --mission
earthcare` into `figures/earthcare/` and shown in `15_earthcare_results.md`.

## Figures

| # | content | source | status |
|---|---|---|---|
| Fig. 1 | Concept, one panel per mission: passive swath and radar footprint (CPR nadir line; PR/PMR scanned swath), one-minute decision segments | schematic | ☐ draw |
| Fig. 2 | Map of decisions coloured by radar-footprint label, one panel per mission (shows TRMM's tropical band and EarthCARE's polar coverage) | new parquets | ◐ `fig2_rain_map.png` (EarthCARE; 5° cells, no coastlines) |
| Fig. 3 | **Main result:** capture vs. duty cycle for all policies, with random diagonal and oracle curve | E3 | ◐ `fig3_capture_curves.png` (EarthCARE) |
| Fig. 4 | Radar footprint vs. passive swath: positive-rate bars and capture gap, per mission | E4 | ◐ `fig4_label_geometry.png` (EarthCARE) |
| Fig. 5 | Capture vs. duty cycle per feature tier, night only (or day/night panels) | E5 | ◐ `fig5_tiers_day_night.png` (EarthCARE) |
| Fig. 6 | Skill by latitude band and surface type | E6 | ◐ `fig6_strata.png` (EarthCARE) |
| Fig. 7 | Mission comparison: capture@d per mission at shared budgets; TRMM AUC by local hour with EarthCARE's two sampling times marked; TRMM rain by local hour | E12 | ◐ `comparison/fig7_mission_comparison.png` (EarthCARE, TRMM) |
| Fig. S1 | Bootstrap intervals per policy | E10 | ◐ `figS1_bootstrap.png` (EarthCARE) |
| Fig. S2 | Feature importance for best policy | E11 | ◐ `figS2_feature_importance.png` (EarthCARE) |
| Fig. S3 | AUC by held-out month for P5–P7 vs. each model's summer-trained time-blocked test | E2, E3 | ◐ `figS3_split_robustness.png` (EarthCARE) |
| Fig. S4 | Autocorrelation of the label and key features vs. lag, per mission | new parquets | ◐ notebook §5 (EarthCARE) |

Numbering: this table is the only source of figure numbers. Every other file refers to figures
as "Fig. N" or "Fig. SN" from this table, and file names follow it (`figN_*.png`, `figSN_*.png`).

Style: one colour per policy, kept identical across Fig. 3 and Fig. 5; random and oracle as
grey dashed reference lines; axes in percent.

## Tables

| # | content | source | status |
|---|---|---|---|
| Table 1 | Positioning vs. dynamic targeting | `02_related_work.md` §2.6 | ☑ |
| Table 2 | Mission and dataset summary: NORAD ID, orbit, radar footprint, passive sensor and swath, TLE set, rows | `03_dataset.md` §3.6–3.8 | ◐ EarthCARE filled |
| Table 3 | Label positive rates, radar footprint vs. passive swath, per mission | `04_label_finding.md` | ◐ EarthCARE (§4.3) |
| Table 4 | Policy results: AUC, AP, capture@{5,10,13,20,30,50}%, mass-capture | E3 | ◐ EarthCARE (`07_experiments.md` E3) |
| Table 5 | Feature tiers × day/night + policy cost | E5, E9 | ◐ EarthCARE (`06_onboard_feasibility.md`) |
| (supp.) | Full column registry, per mission | `missions/<mission>/column_registry.yaml` | ☑ EarthCARE |
| (supp.) | Random vs. time-blocked split; leave-one-month-out | E1, E2 | ◐ EarthCARE |
| (supp.) | Label-threshold sensitivity | E7 | ◐ EarthCARE |
| Table 6 | Cross-mission comparison and transfer matrix | E12 | ◐ `e12_comparison.csv`, `e12_transfer.csv` (EarthCARE, TRMM) |

## Generation rule

Every figure and table is produced by the baseline script from the parquet, with the parquet hash
written into the figure metadata or caption footnote. No hand-edited numbers.

## Primary seasonal figures (separate from historical numbering)

Generate via `python -m baselines.paper_figures --protocol seasonal --mission <mission>`. All
values come from generated result CSVs, with separate EarthCARE/TRMM figures under
`figures/<mission>/seasonal_lomo/`. The generated report supplies the corresponding captions.

| file | caption and source |
|---|---|
| `monthly_auc.png` | AUC by held-out month and P1–P8; oracle marked as reference. `e2_monthly.csv`. |
| `monthly_capture.png` | Capture at nominal 20% budget per month; read alongside realized duty. `e2_monthly.csv`. |
| `seasonal_capture_curves.png` | One panel per month, all policies; nominal budget on x-axis. `e3_capture_curves.csv`. |
| `chronological_vs_seasonal.png` | Historical pooled stress test versus equal-weight seasonal mean and min–max; different estimands, not a causal penalty. Historical E3 and seasonal E2 summary. |
| `monthly_base_rate_and_duty.png` | Monthly primary-label base rate and realized held-out duty at nominal 20%. `e2_monthly.csv`. |
| `monthly_feature_importance.png` | Fixed P6 permutation AUC loss by feature and month. `e11_permutation_importance.csv`. |

The primary tables are generated in `16_seasonal_results.md`: every monthly score and equal-weight
mean, sample SD and range. `e10_bootstrap.csv` supplies within-month whole-day intervals; do not
label across-fold interval-endpoint summaries as a pooled confidence interval.
