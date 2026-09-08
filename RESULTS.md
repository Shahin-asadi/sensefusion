# Public worked examples — 0.5.2

Inputs were reconstructed from the documented licensed public sources. Core numerical fields match the supplied baseline within the stated tolerance. These examples evaluate the specified datasets and sample structures.

## olive_fusion.csv

[Concise HTML](benchmarks/reference_run/olive_fusion/report.html) · [Print PDF](benchmarks/reference_run/olive_fusion/summary.pdf) · [Complete appendix](benchmarks/reference_run/olive_fusion/extended_report.html)

Input: olive_fusion.csv; 240 rows. Computation completed; evidence availability: complete.

Target: Accelerated ageing duration at 60 C (days). Errors use held-out physical groups; comparisons use the stated common cohort.

all methods: lowest displayed group RMSE = 8.90911 for Late fusion (equal weights), on 240 rows / 24 groups. This ranking is descriptive and may change on new data.

## temperature_bands.csv

[Concise HTML](benchmarks/reference_run/temperature_bands/report.html) · [Print PDF](benchmarks/reference_run/temperature_bands/summary.pdf) · [Complete appendix](benchmarks/reference_run/temperature_bands/extended_report.html)

Input: temperature_bands.csv; 9 rows. Computation completed; evidence availability: complete.

Target: Acetaminophen content (% w/w). Errors use held-out physical groups; comparisons use the stated common cohort.

all methods: lowest displayed group RMSE = 7.3448 for Single block: HIGH, on 9 rows / 9 groups. This ranking is descriptive and may change on new data.
