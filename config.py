"""Configuration constants for the Experiment Architect.

Thresholds in this file are decisions, not facts. Each block names the
convention or citation that justifies the chosen value. Override per-call
when the stakes warrant tighter or looser thresholds.
"""

from typing import Literal

# Statistical convention
# Z_ALPHA is the two-sided 95% critical value used in the sample-size and
# confidence-interval formulas. ALPHA is the matching two-sided significance
# threshold. The two are paired: changing one without the other produces
# inconsistent power and CI behavior.
Z_ALPHA = 1.96  # two-sided 95%, equivalent to norm.ppf(0.975)
Z_BETA = 0.84  # 80% power, equivalent to norm.ppf(0.80)
ALPHA = 0.05  # two-sided significance threshold

# Bayesian sampling
# The posterior win probability and expected loss are estimated by Monte Carlo.
# At 100k draws the standard error on a probability near 0.5 is
# sqrt(0.25 / 100_000) ~= 0.0016, which is well inside the decision thresholds
# below. The seed makes the recommendation reproducible: the same input counts
# always yield the same ship/hold call, which matters for a decision tool.
BAYESIAN_SAMPLES = 100000
BAYESIAN_RANDOM_SEED = 12345

# Bootstrap confidence intervals
# Percentile bootstrap for relative lift on continuous outcomes. Preferred over
# the normal approximation when samples are small or heavily skewed (revenue,
# session length). 5,000 resamples give a stable 95% interval without making CI
# runtime noticeable; the seed keeps the interval reproducible across reruns.
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_RANDOM_SEED = 12345
# Per-group size at or below which the app recommends the bootstrap interval
# over the normal-approximation interval.
SMALL_SAMPLE_THRESHOLD = 30

# Bayesian shipping rule
# These probability cutoffs follow the convention used in the B2C product
# experimentation literature (Beta-Bandit, Convoy, Stitch Fix). Tighter
# cutoffs (e.g., 0.99) are appropriate for irreversible decisions; looser
# ones (0.80) for low-stakes UI tweaks. Override per-call when needed.
BAYESIAN_SHIP_THRESHOLD = 0.95
BAYESIAN_CONSIDER_THRESHOLD = 0.75
BAYESIAN_KEEP_CONTROL_THRESHOLD = 0.25

# Loss tolerance expressed as a fraction of the control conversion rate.
# A relative tolerance handles the wide range of baselines this tool sees
# (1% checkout conversion vs 60% landing-page CTR) without forcing the
# caller to re-tune the absolute threshold each time. The 5% default means
# "no more than 5% of baseline conversions lost in expectation if we ship
# the wrong variant."
DEFAULT_LOSS_TOLERANCE_RELATIVE = 0.05

# Absolute fallback when no baseline is supplied. Kept for backward
# compatibility with callers that pass loss_tolerance directly.
DEFAULT_LOSS_TOLERANCE_ABSOLUTE = 0.005

# Parallel-trends pre-test
# p > 0.10 (not 0.05) intentionally errs toward accepting the null because
# rejecting parallel trends blocks the entire DiD analysis. Roth et al.
# (2023, "Pre-Trends in Difference-in-Differences") note that pre-trends
# tests have low power against the violations they aim to catch, and
# recommend treating non-rejection as "no evidence against" rather than
# "evidence for" parallel trends.
PARALLEL_TRENDS_PASS_THRESHOLD = 0.10

# RDD diagnostics
# Density-ratio bounds for the sorting check around the cutoff. Wider than
# McCrary's optimal-bandwidth procedure because we use a fixed 20% window
# (RDD_DENSITY_WINDOW_FRAC * std) rather than an optimized one.
RDD_DENSITY_RATIO_LOWER = 0.7
RDD_DENSITY_RATIO_UPPER = 1.4
RDD_DENSITY_WINDOW_FRAC = 0.20  # window = this * std(running_var)

# Maximum coefficient shift across the bandwidth sweep before flagging
# instability. 30% follows the Imbens-Lemieux (2008) convention for
# practical RDD sensitivity reporting.
RDD_COEFFICIENT_STABILITY_THRESHOLD = 0.30

# Minimum observations per side of the cutoff for the rule-of-thumb
# bandwidth floor. Below this, the local OLS becomes unstable.
RDD_MIN_SIDE_OBSERVATIONS = 30

# Streamlit config
PAGE_TITLE = "Experiment Architect"
PAGE_ICON = "Build"
PAGE_LAYOUT: Literal["centered", "wide"] = "wide"

# LLM Models
# Update these when the provider releases a better default. The app does not
# hardcode the version anywhere else — only here.
MODEL_OPENAI = "gpt-4.1"
MODEL_ANTHROPIC = "claude-sonnet-4-6"
MODEL_GEMINI = "gemini-2.0-flash"

# LLM Config
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
