# 11. Figures and Tables Plan

Aim for 5–6 figures and 4–5 tables in a full paper; a 4-page conference paper (e.g., IGARSS) keeps
Fig. 1, Fig. 3, Fig. 4, Table 2 and Table 3 only.

## Figures

| # | content | source | status |
|---|---|---|---|
| Fig. 1 | Concept: spacecraft, 150 km MSI swath, nadir CPR line, one-minute decision segments; nadir vs. swath label illustrated on one real decision | schematic + one row from the parquet | ☐ draw |
| Fig. 2 | Global map of decisions coloured by nadir label (one month or downsampled full run) | existing CZML/kepler exports or a cartopy map | ◐ data exists |
| Fig. 3 | **Main result:** capture vs. duty cycle for all policies, with random diagonal and oracle curve | E3 | ☐ |
| Fig. 4 | Nadir vs. swath: positive-rate bars and capture gap from the 2 × 2 grid | E4 | ◐ rates done |
| Fig. 5 | Capture vs. duty cycle per feature tier, night only (or day/night panels) | E5 | ☐ |
| Fig. 6 | Skill by latitude band and surface type | E6 | ☐ |
| (supp.) | Autocorrelation of `prectot_nadir`, `preccon_nadir`, `tautot_mean` vs. lag | `task_outputs/t3_autocorr_report.json` | ☑ data |
| (supp.) | Feature importance for best policy | E11 | ☐ |

Style: one colour per policy, kept identical across Fig. 3 and Fig. 5; random and oracle as
grey dashed reference lines; axes in percent.

## Tables

| # | content | source | status |
|---|---|---|---|
| Table 1 | Positioning vs. dynamic targeting | `02_related_work.md` §2.6 | ☑ |
| Table 2 | Dataset summary: rows, period, resolution, geometry, gaps, column tiers | `03_dataset.md` | ☑ |
| Table 3 | Label positive rates, nadir vs. swath | `04_label_finding.md` §4.3 | ☑ |
| Table 4 | Policy results: AUC, AP, capture@{5,10,13,20,30,50}%, mass-capture | E3 | ☐ |
| Table 5 | Feature tiers × day/night + policy cost | E5, E9 | ☐ |
| (supp.) | Full column registry | `../column_registry.yaml` | ☑ |
| (supp.) | Random vs. time-blocked split; leave-one-month-out | E1, E2 | ☐ |
| (supp.) | Label-threshold sensitivity | E7 | ☐ |

## Generation rule

Every figure and table is produced by the baseline script from the parquet, with the parquet hash
written into the figure metadata or caption footnote. No hand-edited numbers.
