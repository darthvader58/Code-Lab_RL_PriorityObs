# 11. Figures and Tables Plan

Aim for 6–7 figures and 5–6 tables in a full paper; a 4-page conference paper (e.g., IGARSS) keeps
Fig. 1, Fig. 3, Fig. 4, Table 2 and Table 3 only, for EarthCARE. Multi-panel figures get one panel per
mission (EarthCARE, TRMM, GPM) with identical axes. Status ◐ = EarthCARE panel exists (in
`missions/earthcare/results/` or the notebook `missions/earthcare/EarthCare_trackwise.ipynb`), other
missions to come.

## Figures

| # | content | source | status |
|---|---|---|---|
| Fig. 1 | Concept, one panel per mission: passive swath and radar footprint (CPR nadir line; PR/DPR scanned swath), one-minute decision segments | schematic | ☐ draw |
| Fig. 2 | Map of decisions coloured by radar-footprint label, one panel per mission (shows TRMM's tropical band and EarthCARE's polar coverage) | new parquets; cartopy | ◐ notebook §4 (EarthCARE) |
| Fig. 3 | **Main result:** capture vs. duty cycle for all policies, with random diagonal and oracle curve | E3 | ◐ `fig3_capture_curves.png` (EarthCARE) |
| Fig. 4 | Radar footprint vs. passive swath: positive-rate bars and capture gap, per mission | E4 | ◐ data ready (EarthCARE); figure to draw |
| Fig. 5 | Capture vs. duty cycle per feature tier, night only (or day/night panels) | E5 | ◐ `fig5_tiers_night.png` (EarthCARE) |
| Fig. 6 | Skill by latitude band and surface type | E6 | ◐ notebook E6 (EarthCARE) |
| Fig. 7 | Mission comparison: capture@d per mission at shared budgets; TRMM capture by local hour with EarthCARE's two sampling times marked | E12 | ☐ |
| (supp.) | Autocorrelation of the label and key features vs. lag, per mission | new parquets | ◐ notebook §5 (EarthCARE) |
| (supp.) | Feature importance for best policy | E11 | ◐ notebook E11 (EarthCARE) |

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
| (supp.) | Full column registry | `../column_registry.yaml` | ☑ |
| (supp.) | Random vs. time-blocked split; leave-one-month-out | E1, E2 | ◐ EarthCARE |
| (supp.) | Label-threshold sensitivity | E7 | ◐ EarthCARE |
| Table 6 | Cross-mission comparison and transfer matrix | E12 | ☐ |

## Generation rule

Every figure and table is produced by the baseline script from the parquet, with the parquet hash
written into the figure metadata or caption footnote. No hand-edited numbers.
