# 14. Submission Plan

Deadlines change every year. Check each venue's current call for papers before planning around it;
no dates are given here on purpose.

## Candidate venues

| venue | format | fit | notes |
|---|---|---|---|
| IGARSS (IEEE) | 4-page paper | remote sensing + ML; DT work appears in this community | keep Fig. 1, 3, 4 and Tables 2–3 |
| AGU Fall Meeting / EGU General Assembly | abstract + poster/talk | atmospheric/Earth-observation audience; low barrier; good first exposure | abstract from `00_abstract.md` |
| Climate Change AI workshop (NeurIPS / ICLR) | short workshop paper | ML audience; benchmark framing fits | emphasise dataset + leakage + metric |
| *Earth System Science Data* / *Scientific Data* | data descriptor (journal) | the dataset as the contribution | needs DOI, full documentation, usage examples |
| *Atmospheric Measurement Techniques* / IEEE TGRS / JSTARS | full journal paper | strongest home once real-EarthCARE validation is added | likely the follow-up paper, not this one |
| FURI / university symposium | poster | internal milestone | can reuse the AGU/EGU poster |

Suggested order: university symposium or AGU/EGU abstract (early feedback) → IGARSS or CCAI workshop
paper (this baseline paper) → journal paper with real-data validation and/or RL.

## Steps to submission

1. **Run E1 (time-blocked split).** Everything else depends on it.
2. Write `run_baselines.py` producing every table/figure in `11_figures_and_tables.md`.
3. Run E2–E11; fill `07_experiments.md` and `13_key_numbers.md`.
4. Replace every **[TO RUN]** in `00_abstract.md` and `baseline_paper.md`.
5. Write §10 from the template.
6. Verify every reference in `09_references.md` from the original source; re-run the literature
   search for the gap statement in §2.7.
7. Name one target mission with a verified power/duty-cycle figure (§1.1, objection O2).
8. Release: tag the code, deposit parquet + code + `column_registry.yaml` on Zenodo, cite the DOI.
9. Advisor review of the full draft; agree on authorship and acknowledgements (FURI funding, data
   providers: NASA GMAO for G5NR).
10. Format for the venue template in Overleaf.

## Assembling the Overleaf draft

Section order and source files:

| LaTeX section | source |
|---|---|
| Title, abstract, keywords | `00_abstract.md` |
| 1 Introduction | `01_introduction.md` |
| 2 Related work | `02_related_work.md` |
| 3 Dataset | `03_dataset.md` |
| 4 Label geometry | `04_label_finding.md` |
| 5 Problem formulation and baselines | `05_baselines.md` |
| 6 Onboard feasibility | `06_onboard_feasibility.md` |
| 7 Experimental setup | `07_experiments.md` (protocols only; results go to §8) |
| 8 Results and discussion | `10_results_and_discussion.md` + filled tables |
| 9 Limitations and future work | `08_limitations_future_work.md` |
| 10 Conclusion | `08_limitations_future_work.md` §8.4 |
| Data and code availability | Zenodo DOI, repository tag |
| References | `09_references.md` |
| Figures / tables | `11_figures_and_tables.md` |

Tip: `pandoc 03_dataset.md -o 03_dataset.tex` converts each file to LaTeX as a starting point;
tables and math will need light cleanup.
