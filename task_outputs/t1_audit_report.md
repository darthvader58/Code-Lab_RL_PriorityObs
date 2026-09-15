# Task 1 audit: `lwtup_fore`

Audited on 2026-09-14 using only the existing `trackwise_decisions.parquet` and
`task_outputs/t1_lwtup_fore.parquet`; no network access or re-extraction was
performed.

## Result

The artifact is correct and requires no correction. It has 351,639 rows and
the same `decision` sequence as the source. For each row `t`, the audit
recomputed:

```text
expected[t] = lwtup_mean[t + 1] if decision[t + 1] - decision[t] == 1 else NaN
```

The artifact matches this recomputation exactly (all 351,576 finite values,
with no value or mask mismatches). There are 63 NaNs in `lwtup_fore`:

- 50 rows are invalid because the next decision is not consecutive (49 gap
  boundaries plus the final row, which has no next row).
- 13 additional rows point to a consecutive next row whose source
  `lwtup_mean` is NaN.

The source contains 1,161 missing decision IDs in aggregate (`decision`
extends to 352,799 while there are 351,639 rows), represented by 49 observed
non-unit transitions. No gaps were bridged.

Finite-value summary for `lwtup_fore` (n = 351,576): mean 227.372648, standard
deviation 48.160066, minimum 0.000000, median 229.124443, third quartile
267.888809, maximum 383.632202.

This is a geometric forward-looking proxy, not temporal leakage: BBR's fore
view physically observes approximately the region represented by the next
decision. The proxy uses the next MSI-swath-footprint mean; a true BBR fore
view has different footprint geometry and approximately 10 km resolution, so
fidelity should not be overstated.
