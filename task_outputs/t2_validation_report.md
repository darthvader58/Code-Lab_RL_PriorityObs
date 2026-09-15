# Task 2 validation

{
  "nan_counts": {
    "prectot_swath": 26,
    "prectot_nadir": 24,
    "preccon_swath": 10,
    "preccon_nadir": 10,
    "iwp_swath": 0,
    "iwp_nadir": 0,
    "lwp_swath": 0,
    "lwp_nadir": 0
  },
  "zero_counts": {
    "prectot_swath": 5323,
    "prectot_nadir": 9204,
    "preccon_swath": 183398,
    "preccon_nadir": 209331,
    "iwp_swath": 25816,
    "iwp_nadir": 37026,
    "lwp_swath": 13257,
    "lwp_nadir": 19694
  },
  "corrupt_counts": {
    "iwp_nadir": 6,
    "iwp_swath": 9,
    "lwp_nadir": 3,
    "lwp_swath": 6
  },
  "corrupt_total": 24,
  "invalid_entries_all_targets": 94,
  "unique_decisions_any_invalid_target": 51,
  "finite_corrupt_entries": 24,
  "unique_decisions_with_finite_corrupt_values": 15,
  "validity_flags_match_source_rule": true,
  "rule": "finite target with abs(value) >= 1e6 is invalid; invalid values become NaN in final merge; no imputation"
}
