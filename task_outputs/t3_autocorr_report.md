# Autocorrelation diagnostic

Only finite pairs satisfying `decision[i]-decision[i-k] == k` were used.

## `preccon_nadir`

| lag | valid pairs | ACF |
|---:|---:|---:|
| 1 | 351575 | 0.116024 |
| 2 | 351522 | 0.056602 |
| 3 | 351469 | 0.038470 |
| 5 | 351395 | 0.024907 |
| 10 | 351224 | -0.003190 |
| 30 | 350653 | -0.016144 |

Sampled e-folding lag: **1**.

## `prectot_nadir`

| lag | valid pairs | ACF |
|---:|---:|---:|
| 1 | 351555 | 0.171179 |
| 2 | 351499 | 0.060293 |
| 3 | 351441 | 0.015995 |
| 5 | 351369 | 0.011164 |
| 10 | 351199 | 0.005938 |
| 30 | 350628 | -0.007796 |

Sampled e-folding lag: **1**.

## `tautot_mean`

| lag | valid pairs | ACF |
|---:|---:|---:|
| 1 | 351557 | 0.628451 |
| 2 | 351501 | 0.370135 |
| 3 | 351445 | 0.229986 |
| 5 | 351367 | 0.066772 |
| 10 | 351194 | -0.054396 |
| 30 | 350623 | -0.030714 |

Sampled e-folding lag: **3**.

Precipitation persistence is weak at one-minute spacing; this does not prove every sequential feature is useless. A 10-second cadence is a possible future experiment, not tested here.
