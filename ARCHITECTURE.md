# Architecture

## Repository shape

```text
experiment-architect/
├── app.py
├── config.py
├── examples/
│   └── sample_ab_test.csv
├── llm/
│   ├── client.py
│   └── providers.py
├── stats/
│   ├── bayesian.py
│   ├── causal.py
│   ├── frequentist.py
│   ├── plots.py
│   ├── sanity.py
│   └── validation.py
├── tests/
│   ├── test_bayesian.py
│   ├── test_causal.py
│   ├── test_frequentist.py
│   ├── test_llm_client.py
│   └── test_validation.py
├── ui/
│   └── components.py
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Design intent

The app keeps three concerns separate:

1. `app.py` owns user interaction.
2. `stats/` owns calculations, diagnostics, and input contracts.
3. `llm/` owns provider setup plus the small amount of retry logic needed for JSON mapping.

That split matters because it keeps the statistical code testable without Streamlit and keeps the UI from turning into a second analytics layer.

## Request flow

### A/B test design

The design tab uses deterministic code only:

- `stats.frequentist.calculate_sample_size`
- `stats.frequentist.calculate_reverse_mde`
- `stats.sanity.run_all_checks`
- `stats.plots.plot_power_curve`

The sensitivity expander stays in the UI layer, but it only calls those helpers and renders the outputs.

### Raw CSV analysis

This path has four stages:

1. The LLM maps semantic roles such as `variant_col` and `metric_col`.
2. `llm.client.ask_agent_json()` retries once if the payload is malformed or missing required keys.
3. `stats.validation` checks that mapped columns exist, coerces numeric or binary fields, and drops rows missing required values.
4. The app routes the cleaned data to frequentist or Bayesian helpers, then applies frequentist guardrails when the user marks multiple primary metrics or early peeking.

The important boundary is that the LLM never computes the statistic. It only proposes a schema.

### Causal analysis

The causal tabs follow the same pattern:

1. The LLM maps the columns.
2. `stats.validation` enforces binary treatment flags and numeric outcomes.
3. `stats.causal` runs DiD or RDD.
4. The UI surfaces diagnostics instead of hiding them.

## Statistical modules

### `stats/frequentist.py`

Contains:

- chi-squared test for binary outcomes
- Welch's t-test for continuous outcomes
- confidence intervals on relative lift
- sample ratio mismatch check
- Bonferroni-adjusted alpha helper for multiple primary metrics
- peeking and multiple-comparison guardrail summary
- sample size and reverse-MDE calculations

The reverse-MDE helper uses the same split-factor logic as the sample-size helper, so the two calculations stay aligned.

### `stats/bayesian.py`

Implements a Beta-Binomial model for binary metrics. It returns:

- posterior win probability
- expected loss if you ship the variant and it is worse
- posterior alpha/beta parameters for both groups

The decision helper is intentionally loss-aware. High win probability is not enough when the downside remains large.

### `stats/causal.py`

Implements:

- Difference-in-Differences with clustered standard errors by unit
- a pre-period parallel-trends check
- Regression Discontinuity with a density ratio diagnostic near the cutoff
- a rule-of-thumb local bandwidth selector with a bandwidth sweep for robustness
- a small method selector for DiD, RDD, PSM, and CausalImpact

This module is where the project makes its strongest analytical claim: the estimators return diagnostics, not just coefficients.

### `stats/validation.py`

This module exists because the LLM path needed a real schema contract.

It handles:

- mapped-column existence checks
- metric-type normalization
- binary and numeric coercion
- dropping rows missing required fields
- pre-analysis validation for A/B, DiD, and RDD inputs

Without this layer, a plausible-looking but wrong dtype could silently poison the result.

## LLM layer

### `llm/providers.py`

Defines three provider adapters with a shared `call()` interface:

- OpenAI
- Anthropic
- Gemini

### `llm/client.py`

Handles:

- optional dependency checks
- provider selection
- API key lookup from env or Streamlit secrets
- plain text calls
- JSON calls with one retry and a stricter follow-up prompt

This is a thin layer on purpose. It is meant to reduce operational noise, not become an agent framework.

## UI layer

`ui/components.py` keeps rendering code out of `app.py`. These helpers are intentionally small and do not own statistical decisions.

## Tests

The test suite mixes unit tests and simulated-data checks.

- `test_frequentist.py` checks algebra, edge cases, and chi-squared diagnostics.
- `test_bayesian.py` checks posterior behavior and the loss-aware decision rule.
- `test_causal.py` uses synthetic data with known effects and includes placebo and manipulation-style checks.
- `test_llm_client.py` checks JSON retry behavior without calling a real provider.
- `test_validation.py` checks the schema and dtype contract for mapped dataframes.

For repo hygiene, `.github/workflows/tests.yml` runs the test suite on GitHub Actions with Python 3.11.

## Known limits

- LLM column mapping is schema-validated, not semantically guaranteed.
- DiD uses a useful pre-trend warning, but passing that test does not prove identification.
- RDD uses a rule-of-thumb local bandwidth and sweep diagnostics rather than a formal optimal bandwidth estimator.
- Continuous-metric analysis still assumes mean-based summaries are sensible; heavily skewed revenue can need extra work.
- The app exposes Bonferroni-style multiple-comparison guardrails and an early-peeking warning, but not a full sequential-testing framework.
