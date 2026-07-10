# Experiment Architect

[![CI](https://github.com/tomasz-solis/experiment-architect/actions/workflows/tests.yml/badge.svg)](https://github.com/tomasz-solis/experiment-architect/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **Know whether an experiment can answer your question before you spend the traffic on it. And what to do when it can't.**

**[▶ Live demo](https://testarchitect.streamlit.app/):** open it, run the **design** tab for a
reverse-MDE read on a test before you launch it, or upload the bundled sample CSV in the **Raw
CSV audit** lens for a full analysis with a loss-aware ship/hold call. No install. (Hosted on
Streamlit Community Cloud; if it has gone to sleep, give it a few seconds to wake.)

**Run it locally:** `streamlit run app.py` for the interactive app · `uv run --extra dev pytest`
to run the suite.

Experiment Architect is a Streamlit app for planning and reading product experiments. It
covers A/B test sizing, causal method selection, CSV-based result analysis, and a small LLM
layer that maps messy column names before Python runs the statistics. (The app is hosted at
`testarchitect.streamlit.app`.)

If you are evaluating this as a portfolio sample, read [NARRATIVE.md](NARRATIVE.md) first: a
walk-through of the decision this tool is built for, from a doomed A/B to a loss-aware ship
call. For the code structure, see [ARCHITECTURE.md](ARCHITECTURE.md).

## When to reach for this

The tool is built around one judgment most A/B write-ups skip: whether the experiment in
front of you can actually detect the effect you care about in the time you have, and what to
do when it can't. Reach for it when:

- **You are about to run an A/B test** and want the reverse-MDE answer first: what is the
  smallest effect this test could detect given real traffic and baseline rate, before you
  commit the traffic to it.
- **Randomization is not clean** and you need to pick a causal fallback (DiD, RDD) honestly,
  with the diagnostics that say whether the identifying assumptions hold.
- **You want a ship/hold decision framed as expected loss**, not a bare significance verdict:
  a direct probability the variant is better, and the cost if you ship it and you are wrong.

Use something else when you already run a mature in-house experimentation platform with
sequential testing (this is not a full sequential framework), when you need formal causal
identification guarantees (the diagnostics here are warnings, not proofs), or when you need
production metric pipelines rather than CSV-based analysis.

## What it does

- Designs A/B tests with sample size, reverse MDE, split-ratio penalty, and deterministic sanity checks.
- Shows a sensitivity view so you can see how traffic and baseline assumptions change the detectable effect.
- Recommends a causal method when randomization is not possible.
- Runs DiD with a parallel-trends pre-test and clustered standard errors.
- Runs RDD with a density check around the cutoff, a rule-of-thumb local bandwidth, and a bandwidth sweep for robustness.
- Analyzes raw CSVs with frequentist or Bayesian methods after validating mapped columns, dtypes, and missing values.
- Adds frequentist guardrails for multiple primary metrics and early peeking.
- Uses expected loss alongside posterior win probability for Bayesian shipping recommendations.

## How the app is structured

The repo keeps three concerns separate so the statistics stay testable without Streamlit:

- `app.py`: Streamlit UI and orchestration.
- `stats/`: statistical methods, diagnostics, plotting helpers, and input validation.
- `llm/`: provider wrappers plus JSON retry/parsing helpers. The LLM only proposes a column
  schema; it never computes a statistic.
- `ui/`: small, pure rendering helpers, unit-tested without Streamlit.
- `tests/`: unit tests and simulation tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the fuller walkthrough, including the request flow
per analysis path and the type contracts.

## Installation

```bash
git clone https://github.com/tomasz-solis/experiment-architect.git
cd experiment-architect
pip install -r requirements.txt
```

If you want to run tests:

```bash
pip install -r requirements-dev.txt
```

Create a `.env` file if you want AI-assisted column mapping:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# or
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# or
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...
```

Then start the app:

```bash
streamlit run app.py
```

The app still works without an API key. The LLM-powered mapping and code-generation paths are simply disabled.

## Streamlit Community Cloud

For Streamlit Community Cloud, put the same keys in the app's `Secrets` settings instead of committing them to GitHub. A matching example lives in `.streamlit/secrets.toml.example`.

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-your-key-here"
```

The app is kept awake by a scheduled workflow in [product-decision-lab](https://github.com/tomasz-solis/product-decision-lab) that visits all three lab apps every six hours.

## Quick demo data

If you want a fast demo without preparing your own file, upload [examples/sample_ab_test.csv](examples/sample_ab_test.csv) in the **Raw CSV audit** lens. It contains 1,000 rows (500 control / 500 treatment), a binary `converted` column, and a continuous `revenue` column, enough to exercise both analysis paths.

## Testing and quality

```bash
uv run --extra dev pytest
```

The suite mixes unit tests and simulated-data checks:

- frequentist helpers (algebra, edge cases, chi-squared diagnostics)
- Bayesian decision logic (posterior behavior and the loss-aware ship/hold rule)
- causal simulation checks (synthetic data with known effects, plus placebo and manipulation checks)
- LLM JSON retry logic (without calling a real provider)
- dataframe validation contracts

Beyond the tests, the whole repo (`app.py`, `ui/`, `stats/`, `llm/`, and `tests/`) is checked
under `mypy --strict` in CI, statistical helpers return `TypedDict` result shapes rather than
loose dicts, and `tests/test_app_smoke.py` runs the real Streamlit script end-to-end through
`AppTest`. CI (`.github/workflows/tests.yml`) runs the suite on Python 3.11.

## Notes on method choice

- Use the frequentist path when you need a familiar significance test and confidence interval.
- If you reviewed several primary metrics, use the adjusted alpha shown in the app rather than treating every p-value as if it came from a single test.
- If you peeked before the planned stop date, treat the frequentist result as directional unless you used a sequential design.
- Use the Bayesian path when you want a direct probability statement and a loss-aware ship/hold decision.
- For continuous metrics with heavy right skew, the mean-based Welch test can be brittle. For small groups (30 or fewer observations) the app automatically switches to a percentile bootstrap confidence interval and an unequal-variance effect size; for larger skewed samples a log transform is still worth considering before you trust the result.

## Limitations

- Column mapping is validated against the dataframe schema, but semantic correctness is still on you. A model can pick a real column for the wrong role.
- DiD and RDD surface diagnostics, but those checks are warnings, not proofs of identification.
- The app includes Bonferroni-style multiple-comparison guardrails and a peeking warning, but it does not implement a full sequential-testing design.
- The RDD bandwidth selector is a transparent rule of thumb, not an optimal bandwidth estimator.
- PSM and CausalImpact remain code-generation paths, not in-app estimators.

## License

MIT. See [LICENSE](LICENSE).

## Part of the Product Decision Lab

Experiment Architect is one of three headline projects in my
[Product Decision Lab](https://github.com/tomasz-solis/product-decision-lab) —
measurement readiness, experimentation, and decision analysis for product teams deciding
under uncertainty.

Tomasz Solis — [LinkedIn](https://www.linkedin.com/in/tomaszsolis) · [GitHub](https://github.com/tomasz-solis)
