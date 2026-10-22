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

## Mission staging and open decisions

Missions are done one at a time, in full depth, in this order (advisor's recommendation: one case in
detail over three at surface level):

| stage | mission | gate to move on |
|---|---|---|
| 1 | **EarthCARE** ☑ | season-aligned extraction complete (352,800 rows, no gaps); E1–E11 done (18 Oct 2026) |
| 2 | **TRMM** ◐ | TLE set received ☑; code ready and checked offline ☑ (22 Oct 2026); dataset built with PR-swath labels and VIRS-swath features ☐; E1–E11 done ☐ |
| 3 | **GPM** | decisions D2 and D3 made; dataset built; E1–E11 done |
| 4 | comparison | E12 on the completed missions |

Open decisions:

| ID | decision | options | recommendation |
|---|---|---|---|
| D1 | paper scope at submission time | three missions / EarthCARE + TRMM / EarthCARE only | submit with every mission that has reached stage gate; move the rest to a follow-up |
| D2 | GPM passive data source | real GPM data / microwave forward model on G5NR / G5NR proxies | real GPM data (`03_dataset.md` §3.8) |
| D3 | GPM label | added structure / precipitation GMI misses / GMI retrieval uncertainty | decide with advisor before building GPM (`05_baselines.md` §5.8) |
| D4 | TRMM passive context | VIRS only / VIRS + TMI | VIRS only first; TMI variant only if D2 provides brightness temperatures |

**Fallback.** If only EarthCARE reaches its stage gate, submit the EarthCARE-only version: title
option 2 in `00_abstract.md`, drop RQ5, contribution 5, E12, §10.8, Fig. 7, and Table 6, and keep TRMM
and GPM as future work in §8.3. The rest of the paper is unchanged.

**TLE requests** (CelesTrak special data request, format TLE): EarthCARE 59908, 2025-05-25 →
2026-02-08; TRMM 25063, 2005-05-25 → 2006-02-08; GPM 39574, 2025-05-25 → 2026-02-08.

## Steps to submission

1. ☑ **Finish the EarthCARE extraction**, then run E1 (time-blocked split).
2. ☑ Write `run_baselines.py --mission …` producing every table/figure in `11_figures_and_tables.md`.
3. ☑ Run E2–E11 for EarthCARE; fill `07_experiments.md` and `13_key_numbers.md`. Hardware timing
   (Raspberry Pi–class or flight-representative board) is deferred: Limitation 11, future work 7.
3a. Repeat stages 2–4 for TRMM and GPM as time allows; decide D1 before writing.
4. Replace every **[TO RUN]** in `00_abstract.md` and `baseline_paper.md`.
5. Write §10 from the template.
6. Verify every reference in `09_references.md` from the original source; re-run the literature
   search for the gap statement in §2.7.
7. Name one target mission with a verified power/duty-cycle figure (§1.1, objection O2).
8. Release: tag the code, deposit each mission's parquet, its TLE set, code, and
   `column_registry.yaml` on Zenodo, cite the DOI.
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
| 3 Datasets (shared method + one subsection per mission) | `03_dataset.md` |
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
