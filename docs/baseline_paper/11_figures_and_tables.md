# 11. Figures and Tables Plan

Aim for 6–7 figures and 5–6 tables in a full paper; a 4-page conference paper (e.g., IGARSS) keeps
Fig. 1, Fig. 3, Fig. 4, Table 2 and Table 3 only, for EarthCARE. Multi-panel figures get one panel per
mission (EarthCARE, TRMM, GPM) with identical axes. All "data exists" statuses below referred to the
deleted earlier EarthCARE extraction and are reset.

## Figures

| # | content | source | status |
|---|---|---|---|
| Fig. 1 | Concept, one panel per mission: passive swath and radar footprint (CPR nadir line; PR/DPR scanned swath), one-minute decision segments | schematic | ☐ draw |
| Fig. 2 | Map of decisions coloured by radar-footprint label, one panel per mission (shows TRMM's tropical band and EarthCARE's polar coverage) | new parquets; cartopy | ☐ |
| Fig. 3 | **Main result:** capture vs. duty cycle for all policies, with random diagonal and oracle curve | E3 | ☐ |
| Fig. 4 | Radar footprint vs. passive swath: positive-rate bars and capture gap, per mission | E4 | ☐ |
| Fig. 5 | Capture vs. duty cycle per feature tier, night only (or day/night panels) | E5 | ☐ |
| Fig. 6 | Skill by latitude band and surface type | E6 | ☐ |
| Fig. 7 | Mission comparison: capture@d per mission at shared budgets; TRMM capture by local hour with EarthCARE's two sampling times marked | E12 | ☐ |
| (supp.) | Autocorrelation of the label and key features vs. lag, per mission | new parquets | ☐ |
| (supp.) | Feature importance for best policy | E11 | ☐ |

Style: one colour per policy, kept identical across Fig. 3 and Fig. 5; random and oracle as
grey dashed reference lines; axes in percent.

## Tables

| # | content | source | status |
|---|---|---|---|
| Table 1 | Positioning vs. dynamic targeting | `02_related_work.md` §2.6 | ☑ |
| Table 2 | Mission and dataset summary: NORAD ID, orbit, radar footprint, passive sensor and swath, TLE set, rows | `03_dataset.md` §3.6–3.8 | ◐ layout done |
| Table 3 | Label positive rates, radar footprint vs. passive swath, per mission | `04_label_finding.md` | ☐ |
| Table 4 | Policy results: AUC, AP, capture@{5,10,13,20,30,50}%, mass-capture | E3 | ☐ |
| Table 5 | Feature tiers × day/night + policy cost | E5, E9 | ☐ |
| (supp.) | Full column registry | `../column_registry.yaml` | ☑ |
| (supp.) | Random vs. time-blocked split; leave-one-month-out | E1, E2 | ☐ |
| (supp.) | Label-threshold sensitivity | E7 | ☐ |
| Table 6 | Cross-mission comparison and transfer matrix | E12 | ☐ |

## Generation rule

Every figure and table is produced by the baseline script from the parquet, with the parquet hash
written into the figure metadata or caption footnote. No hand-edited numbers.
