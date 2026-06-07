# Causal methods: identification assumptions and implementation choices

This doc covers the three analytical choices a reviewer probes when reading
the causal module: why clustered standard errors, why the parallel-trends
test runs the way it does, and why the RDD bandwidth uses a rule of thumb
rather than an optimal estimator.

## DiD: clustered standard errors by unit

The DiD estimator regresses `outcome ~ treated + post + treated:post` and
clusters standard errors by `unit`. Clustering matters when observations
within a unit are correlated, which is the default assumption in panel
data. Without clustering, t-statistics are inflated and false-positive
rates rise to 15-25% rather than the nominal 5% (Bertrand, Duflo,
Mullainathan 2004, "How Much Should We Trust Differences-in-Differences
Estimates?").

The implementation uses statsmodels' `cov_type="cluster"` with
`cov_kwds={"groups": unit_col}`. This is the standard CR1 estimator. CR1
is fine when the number of clusters exceeds ~40; below that, consider
wild-cluster bootstrap (Cameron, Gelbach, Miller 2008) or CR2.

## Parallel-trends pre-test

The pre-period model is `outcome ~ treated + time + treated:time` fit on
pre-intervention rows only. The interaction p-value is the test statistic.
We use p > 0.10 (not 0.05) as the "pass" threshold because rejecting
parallel trends is the more costly mistake - it blocks the entire DiD
analysis.

Roth, Sant'Anna, Bilinski, and Poe (2023, "What's Trending in
Difference-in-Differences? A Synthesis of the Recent Econometrics
Literature") note that pre-trends tests have low power against the
violations they aim to catch, and recommend treating non-rejection as
"no evidence against" rather than "evidence for" parallel trends.

When the data has only one pre-period, the test cannot run. The function
returns `passes=True` with `test_ran=False` so the caller can surface the
right warning rather than treating it as a clean pass. This edge case is
covered by `test_parallel_trends_skips_when_only_one_pre_period`.

## RDD: rule-of-thumb local bandwidth

Bandwidth selection is `1.84 * scale * n^(-1/5)`, the
Imbens-Kalyanaraman rule of thumb for sharp RDD. The scale term uses
the smaller of std and IQR/1.349, falling back through median-distance
to 1.0 on pathological data.

This is intentionally not the optimal MSE-minimizing bandwidth (Calonico,
Cattaneo, Titiunik 2014). The CCT procedure requires bias-correction
terms that add complexity beyond what this app aims to provide.
The bandwidth sweep (local / half / full) with the
`coefficient_stable_under_bandwidth` flag is the sensitivity check that
catches the cases where the rule of thumb misleads. The flag fires when
the coefficient shifts by more than `RDD_COEFFICIENT_STABILITY_THRESHOLD`
(default 30%) across the sweep, following the Imbens-Lemieux (2008)
convention.

The density-continuity check uses a 20% window around the cutoff, not
McCrary's optimal bandwidth, because the window-based check is easier to
explain to non-technical reviewers. McCrary's optimal-bandwidth procedure
is the right next step for a production system.

## What the tool does not do

These are real limits of the current implementation. Each one shows up
in `LIMITATIONS.md` as well.

- Passing the parallel-trends test does not prove parallel trends. It
  means we have no statistical evidence against them given the sample.
- The RDD density check catches sorting at the cutoff. It does not catch
  manipulation that preserves the cutoff density (e.g., manipulation on
  a different variable correlated with the running variable).
- DiD assumes no anticipation effects in the pre-period. The tool does
  not test for this.
- Neither method handles staggered treatment timing. For staggered DiD,
  use Callaway-Sant'Anna (2021) or Sun-Abraham (2021), neither of which
  is implemented here.
- The covariance estimator is CR1. Below 40 clusters, switch to CR2 or
  wild-cluster bootstrap.

## Calibration evidence

The `tests/test_calibration.py` suite verifies that the DiD and RDD
estimators perform at their advertised levels on synthetic data. Last
recorded run (N=200 simulations, seed 42):

| Property | Measured | Target |
|---|---|---|
| Chi-squared FPR under null | 2.5% | ~5% |
| Chi-squared power at 5pp lift | 91% | >90% |
| Bayesian ship rate under null | 4% | <10% |
| Parallel-trends power at slope=2 | 100% | >80% |
| Parallel-trends FPR at slope=0 | 9.5% | <20% |

These numbers come from a single seed; they will move with reseeding.
The point of the calibration suite is that the estimators sit inside
their advertised bands, not that any single point estimate is exact.

## References

- Bertrand, Duflo, Mullainathan (2004). "How Much Should We Trust
  Differences-in-Differences Estimates?" QJE 119(1).
- Cameron, Gelbach, Miller (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." Review of Economics and Statistics.
- Roth, Sant'Anna, Bilinski, Poe (2023). "What's Trending in
  Difference-in-Differences?" Journal of Econometrics.
- Imbens, Kalyanaraman (2012). "Optimal Bandwidth Choice for the
  Regression Discontinuity Estimator." Review of Economic Studies.
- Calonico, Cattaneo, Titiunik (2014). Paper on nonparametric confidence
  intervals for regression-discontinuity designs. Econometrica.
- Imbens, Lemieux (2008). "Regression Discontinuity Designs: A Guide to
  Practice." Journal of Econometrics.
- McCrary (2008). "Manipulation of the Running Variable in the
  Regression Discontinuity Design: A Density Test." Journal of
  Econometrics.
- Callaway, Sant'Anna (2021). "Difference-in-Differences with Multiple
  Time Periods." Journal of Econometrics.
- Sun, Abraham (2021). "Estimating Dynamic Treatment Effects in Event
  Studies with Heterogeneous Treatment Effects." Journal of
  Econometrics.
