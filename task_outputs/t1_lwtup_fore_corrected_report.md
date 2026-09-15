# Corrected Task 1 validation

- **rows:** 351639
- **finite_values:** 339859
- **final_row_nans:** 1
- **gap_boundary_nans:** 49
- **weather_step_boundary_nans:** 11717
- **missing_next_lwtup_mean_nans:** 13
- **total_nans:** 11780
- **finite_max_abs_error_vs_formula:** 0.0
- **field_g5_step_transition_disagreements:** 0
- **field_g5_step_transition_agreements:** 351638

`lwtup_fore` is a geometric forward-region proxy for BBR. It uses the next decision's MSI-swath-style GEOS-5 mean only when the next decision is adjacent and remains in the same G5NR weather step; it is not a true BBR retrieval or footprint.
