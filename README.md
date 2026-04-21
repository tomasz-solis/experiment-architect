# Experiment Architect

Experiment Architect is the repo for Test Architect, a Streamlit app for planning and analyzing product experiments. It covers A/B test sizing, causal method selection, CSV-based result analysis, and a small LLM layer that maps messy column names before Python runs the statistics.

Live app: `https://testarchitect.streamlit.app/`

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

The repo follows a simple split:

- `app.py`: Streamlit UI and orchestration.
- `stats/`: statistical methods, diagnostics, plotting helpers, and input validation.
- `llm/`: provider wrappers plus JSON retry/parsing helpers.
- `ui/`: small rendering helpers.
- `tests/`: unit tests and simulation tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the fuller walkthrough.

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

If you want the keep-awake workflow to ping the deployed app, add a repository variable named `STREAMLIT_APP_URL` with your public app URL, for example `https://your-app-name.streamlit.app/`.

## Quick demo data

If you want a fast demo without preparing your own file, upload [examples/sample_ab_test.csv](examples/sample_ab_test.csv) in the `Raw CSV (Automated Analysis)` tab.

## Testing

```bash
uv run --extra dev pytest
```

The current suite covers:

- frequentist helpers
- Bayesian decision logic
- causal simulation checks
- LLM JSON retry logic
- dataframe validation contracts

The repo also includes a lightweight GitHub Actions workflow at `.github/workflows/tests.yml` that installs `requirements-dev.txt` and runs `pytest`.

## Notes on method choice

- Use the frequentist path when you need a familiar significance test and confidence interval.
- If you reviewed several primary metrics, use the adjusted alpha shown in the app rather than treating every p-value as if it came from a single test.
- If you peeked before the planned stop date, treat the frequentist result as directional unless you used a sequential design.
- Use the Bayesian path when you want a direct probability statement and a loss-aware ship/hold decision.
- For continuous metrics with heavy right skew, the mean-based Welch test can be brittle. Revenue often benefits from a log transform or a bootstrap check before you trust the result.

## Limitations

- Column mapping is validated against the dataframe schema, but semantic correctness is still on you. A model can pick a real column for the wrong role.
- DiD and RDD surface diagnostics, but those checks are warnings, not proofs of identification.
- The app includes Bonferroni-style multiple-comparison guardrails and a peeking warning, but it does not implement a full sequential-testing design.
- The RDD bandwidth selector is a transparent rule of thumb, not an optimal bandwidth estimator.
- PSM and CausalImpact remain code-generation paths, not in-app estimators.

## Contact

Tomasz Solis — [LinkedIn](https://linkedin.com/in/tomaszsolis) · [GitHub](https://github.com/tomasz-solis)
