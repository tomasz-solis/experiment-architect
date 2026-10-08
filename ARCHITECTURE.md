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
│   ├── decision_cards.py
│   ├── frequentist.py
│   ├── plots.py
│   ├── power.py
│   ├── prereg.py
│   ├── sanity.py
│   └── validation.py
├── tests/
│   ├── test_app_smoke.py
│   ├── test_bayesian.py
│   ├── test_calibration.py
│   ├── test_causal.py
│   ├── test_components.py
│   ├── test_decision_cards.py
│   ├── test_formatting.py
│   ├── test_frequentist.py
│   ├── test_llm_client.py
│   ├── test_plots.py
│   ├── test_power.py
│   ├── test_prereg.py
│   ├── test_providers.py
│   ├── test_sanity.py
│   └── test_validation.py
├── ui/
│   ├── components.py
│   ├── formatting.py
│   ├── snapshots.py
│   └── state.py
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

### Power, variance, and the pre-registered plan

The planning section is deterministic and self-contained:

- `stats.power.sample_size_continuous` and `mde_from_sample_continuous` size a value metric from its variance rather than a base rate
- `stats.power.cuped_variance_retained` and `estimate_cuped_rho` price variance reduction against a measured pre-period correlation
- `stats.power.design_effect` prices a clustered randomisation (team, market, account) using the Kish formula, and the result is threaded through the sizing above rather than sized separately
- `stats.power.plan_duration` converts a sample requirement into calendar time, including the ramp phase and the maturation window the last cohort still needs
- `stats.power.skew_diagnostics` and `simulate_power` replace the normal approximation with a measurement when the metric is heavily skewed
- `stats.power.intensity_options` and `compliance_effects` cost treatment doses and separate ITT from the complier effect
- `stats.prereg.build_preregistration` freezes the design, including the design effect and a guardrail baseline, in session state
- `stats.prereg.read_guardrail` reads what a guardrail actually did once data arrives, and will not call an underpowered clean reading proof of safety
- `stats.prereg.serialise_preregistration` and `parse_preregistration` download and restore the locked plan as JSON, so it survives a page refresh

The last three are what connects the two halves of the app. Once a plan is locked, the manual and CSV readouts call `stats.prereg.verify_against_plan` and render the delivered sample, split, estimand, transform, alpha spend, and guardrail sensitivity against what was promised, before the effect is shown.

### Raw CSV analysis

This path has five stages:

1. The user picks the variant column, the outcome column, and the outcome type from ordinary dropdowns, defaulted by `stats.validation.guess_mapping()` matching common column names. An optional "Ask the model to suggest the mapping" button pre-fills those dropdowns when a key is configured; the analysis itself never depends on it.
2. `llm.client.ask_agent_json()` retries once if the suggested payload is malformed or missing required keys, and the button says plainly when the model could not read the headers.
3. `stats.validation` checks that the chosen columns exist, coerces numeric or binary fields, drops rows missing required values, and, if the user names a unit column, `check_analysis_unit` warns when the file holds several rows per randomised unit, since the tests below treat every row as independent.
4. `stats.validation.resolve_control_group` picks the control arm by name (`"control"`, `"A"`, `"baseline"`, and similar), falling back to alphabetical order, so row order in the upload never decides the lift direction; the app states which arm it picked. `stats.frequentist.check_srm` runs against the intended split before and after cleaning, and `check_differential_attrition` tests whether the cleaning step itself dropped rows unevenly between arms.
5. The app routes the cleaned data to frequentist or Bayesian helpers, then applies frequentist guardrails when the user marks multiple primary metrics or early peeking.

The important boundary is that the LLM never computes the statistic. It only proposes a schema, and only as a shortcut.

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
- confidence intervals on relative lift (normal approximation and percentile bootstrap)
- sample ratio mismatch check
- Bonferroni-adjusted alpha helper for multiple primary metrics
- peeking and multiple-comparison guardrail summary
- sample size and reverse-MDE calculations

The reverse-MDE helper uses the same split-factor logic as the sample-size helper, so the two calculations stay aligned.

For continuous outcomes, Welch's t-test exposes two effect sizes: pooled-SD Cohen's d (the default) and an `"averaged"` form using `sqrt((var_a + var_b) / 2)`, which is the variance structure Welch itself uses and the consistent choice under unequal variances. When a group has 30 or fewer observations, the app switches to that effect size and a percentile bootstrap CI (`bootstrap_ci_relative_lift_continuous`), which avoids the normal approximation that gets brittle on small or skewed samples.

### `stats/power.py`

Covers the planning mathematics that a binary sample-size formula leaves out:

- continuous-outcome sizing, where the requirement scales with variance rather than a base rate
- CUPED and regression-adjustment variance reduction, including estimating the pre/post correlation from real data
- the allocation penalty an uneven split pays, shared with `stats/frequentist.py` so the two cannot diverge
- the design effect (Kish) that a clustered randomisation costs, threaded through the sample-size, reverse-MDE, and guardrail-sensitivity calculations
- duration planning that separates newly eligible units from daily actives and adds the maturation window of the last enrolled cohort
- treatment-intensity costing with the marginal return per extra effect point
- ITT as the primary estimand plus CACE/LATE via the Wald instrumental-variables ratio
- skew diagnostics and simulation-based power, which resample real historical outcomes and run the exact planned analysis, including any pre-registered winsorisation
- the smallest guardrail regression the test could actually have detected, corrected for the same design effect
- a metric-layer classifier that fails an outcome defined on post-assignment state

The simulation always runs a zero-lift case alongside the powered one. If the false-positive rate under no effect is not close to alpha, the analysis method is miscalibrated on that distribution and any closed-form sample size is unreliable, so the result says so rather than leaving it to the reader.

### `stats/prereg.py`

Holds the pre-registration contract and the verification that carries it into the readout. `build_preregistration` records the decisions most often revised after data arrives, particularly the outcome transform, the number of primary metrics, and the cluster design effect the plan was sized under. `verify_against_plan` returns one row per commitment, ordered so a broken commitment is read before the effect it would otherwise qualify. `read_guardrail` reads what a guardrail actually did, in either direction (a rise or a drop can be the harm, set by `higher_is_worse`), and will not call an underpowered clean reading proof of safety. `summarise_readout` states the result as absolute uplift, relative uplift, interval, and business impact; it separates an inconclusive test from evidence of no effect, and within that, a result that only ruled out a win from one that ruled out a loss too. `serialise_preregistration` and `parse_preregistration` let a locked plan be downloaded and restored, so a page refresh no longer erases it; restoring validates the file rather than trusting it, and a plan saved by an older version of the app loads with documented compatibility defaults.

### `stats/bayesian.py`

Implements a Beta-Binomial model for binary metrics. It returns:

- posterior win probability
- expected loss if you ship the variant and it is worse
- posterior alpha/beta parameters for both groups

The decision helper is intentionally loss-aware. High win probability is not enough when the downside remains large.

The posterior win probability and expected loss are estimated by seeded Monte Carlo (`BAYESIAN_RANDOM_SEED`), so the same input counts always yield the same ship/hold recommendation. That reproducibility matters for a decision tool: two analysts looking at the same data should not see different calls because of sampling noise.

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

`ui/components.py` keeps Streamlit rendering code out of `app.py`. These helpers are intentionally small and do not own statistical decisions.

`ui/formatting.py` holds the *pure* presentation helpers (`first_sentence`, `duration_tone`, `build_card`, `sidebar_tip`). They contain no Streamlit calls, so unlike `app.py` — which executes Streamlit at import time — they can be imported and unit tested directly (`tests/test_formatting.py`).

`ui/state.py` centralizes the Streamlit session-state widget keys as constants plus the `read_uploaded_dataframe` reader. Before this, a key like `"main_base"` was a literal in both the widget definition and the snapshot builder, so a rename could silently desync them.

`ui/snapshots.py` owns the four review lenses. Each builder (`design_snapshot`, `manual_snapshot`, `csv_snapshot`, `causal_snapshot`) reads the relevant session state and returns the hero/summary content for one lens; `build_page_snapshot` dispatches on the selected lens. Pulling these out of `app.py` keeps the app script a thin orchestrator and keeps each lens cohesive. They are exercised end-to-end by `tests/test_app_smoke.py`, which runs the real Streamlit script through `AppTest`.

## Type contracts

The statistical helpers return `TypedDict` results (`ChiSquaredResult`, `WelchTTestResult`, `SampleSizeResult`, `FrequentistGuardrails`, `BayesianAnalysisResult`, …) rather than loosely-typed dicts. This documents each result shape, lets `mypy --strict` verify call sites instead of forcing `float(...)`/`bool(...)` casts, and prevents key typos. The whole repo — `app.py`, `ui/`, `stats/`, `llm/`, and `tests/` — is checked under `mypy --strict` in CI.

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
- The app exposes Bonferroni-style multiple-comparison guardrails and an early-peeking warning, but not a full sequential-testing framework. Unplanned looks are flagged, not corrected with an alpha-spending schedule.
- Simulation-based power resamples the uploaded history, so it inherits whatever selection that sample carries. It answers "is the analysis method calibrated on this shape", not "is this sample representative".
- CACE/LATE assumes the exclusion restriction and no defiers. The app reports it as a labelled secondary to ITT and does not test those assumptions.
- Cluster randomisation is sized with a design effect, but that correction is only as good as the intracluster correlation the user supplies. The app does not estimate it from data.
- Plan verification depends on analyst attestation for the facts the app cannot observe, such as whether the analysis really covered every randomised unit.
- Guardrail direction (whether a rise or a drop counts as harm) is declared by the analyst, not inferred from the metric name.
- The unit-of-analysis check needs the user to name a unit column, and it warns rather than aggregating; it does not fix a clustered file on its own.
- The sample-ratio-mismatch threshold, 0.001, is a convention, chosen because the check runs on every readout and an ordinary 0.05 threshold would cry wolf on routine sampling noise.
- A downloaded plan is a file the user has to keep. Nothing is stored server-side, so losing the file loses the plan.
