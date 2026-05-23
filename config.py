"""Configuration constants for the Experiment Architect.

Thresholds in this file are decisions, not facts. Each block names the
convention or citation that justifies the chosen value. Override per-call
when the stakes warrant tighter or looser thresholds.
"""

# Statistical convention
# Z_ALPHA is the two-sided 95% critical value used in the sample-size and
# confidence-interval formulas. ALPHA is the matching two-sided significance
# threshold. The two are paired: changing one without the other produces
# inconsistent power and CI behavior.
Z_ALPHA = 1.96  # two-sided 95%, equivalent to norm.ppf(0.975)
Z_BETA = 0.84  # 80% power, equivalent to norm.ppf(0.80)
ALPHA = 0.05  # two-sided significance threshold

# Bayesian sampling
BAYESIAN_SAMPLES = 100000

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
PAGE_TITLE = "Test Architect"
PAGE_ICON = "🏗️"
PAGE_LAYOUT = "wide"

# LLM Models
MODEL_OPENAI = "gpt-4o"
MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
MODEL_GEMINI = "gemini-1.5-pro"

# LLM Config
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
